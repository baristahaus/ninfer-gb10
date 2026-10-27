#!/usr/bin/env python3
"""Measure matched Flash-Next HTTP workloads against an already running server."""
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def request(url, messages, limit, started=None, wait_for=None, reuse=False, cache_salt=None):
    if wait_for is not None and not wait_for.wait(300):
        raise TimeoutError('first request did not begin streaming')
    payload = {'model': 'Qwen/Qwen3.8-Flash-Next', 'messages': messages,
               'max_tokens': limit, 'temperature': 0, 'top_p': 1,
               'presence_penalty': 0, 'frequency_penalty': 0, 'stream': True,
               'stream_options': {'include_usage': True},
               'chat_template_kwargs': {'preserve_thinking': True},
               'cache_salt': cache_salt or uuid.uuid4().hex}
    begin = time.monotonic()
    stamps, text, usage = [], [], {}
    reasoning, content_parts = [], []
    req = urllib.request.Request(url + '/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            obj = json.loads(data)
            if obj.get('usage'):
                usage = obj['usage']
            if obj.get('error'):
                raise RuntimeError(obj['error'])
            for choice in obj.get('choices', []):
                delta = choice.get('delta', {})
                thought = delta.get('reasoning_content') or delta.get('reasoning')
                if thought:
                    reasoning.append(thought)
                if delta.get('content'):
                    content_parts.append(delta['content'])
                content = delta.get('content') or thought
                if content:
                    stamps.append(time.monotonic())
                    text.append(content)
                    if started is not None:
                        started.set()
    end = time.monotonic()
    if not stamps or not usage.get('completion_tokens'):
        raise RuntimeError('missing streamed content or completion usage')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    if not reuse and cached:
        raise RuntimeError('cold workload reused a prefix; restart NInfer with --no-prefix-reuse')
    n = usage['completion_tokens']
    assistant = {'role': 'assistant', 'content': ''.join(content_parts)}
    if reasoning:
        assistant['reasoning_content'] = ''.join(reasoning)
    return {'assistant_message': assistant, 'begin': begin, 'end': end, 'first': stamps[0], 'last': stamps[-1],
            'ttft': stamps[0] - begin, 'usage': usage, 'text': ''.join(text),
            'decode_tps': (n - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
            'mean_itl': (stamps[-1] - stamps[0]) / (n - 1) if n > 1 else None,
            'max_stream_gap': max((b-a for a,b in zip(stamps, stamps[1:])), default=0)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--contexts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=5)
    p.add_argument('--tokens', type=int, default=512,
                   help='maximum output tokens per request; natural stops remain enabled')
    p.add_argument('--cases', nargs='+', choices=['single8k','pair8k','single64k','pair64k','mixed','turns'], default=['single8k','pair8k','single64k','pair64k','mixed'])
    a = p.parse_args()
    contexts = json.loads(a.contexts.read_text())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def prompt(length, lane, iteration=None):
        prefix = f'Request {lane}. ' + (f'Conversation {iteration}. ' if iteration is not None else '')
        return [{'role':'user','content':prefix+'Read the following reference material.\n'+contexts[str(length)]+ '\nExplain its main themes in detail, with examples. Write at least 1000 words.'}]
    request(a.url, [{'role':'user','content':'Explain why the sky is blue.'}],32)
    with a.output.open('w') as out:
        for case in a.cases:
            for repetition in range(-1, a.repetitions):
                histories = [prompt(8192,0,repetition),prompt(8192,1,repetition)]
                salts = [uuid.uuid4().hex, uuid.uuid4().hex]
                for turn in range(8 if case == 'turns' else 1):
                    count = 1 if case.startswith('single') else 2
                    length = 65536 if '64k' in case else 8192
                    ready = threading.Event()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                        futures=[]
                        for lane in range(count):
                            messages = histories[lane] if case == 'turns' else prompt(65536 if case=='mixed' and lane else length,lane)
                            futures.append(pool.submit(request,a.url,messages,a.tokens,
                                ready if case=='mixed' and lane==0 else None,
                                ready if case=='mixed' and lane==1 else None,case=='turns',
                                salts[lane] if case=='turns' else None))
                        results=[f.result() for f in futures]
                    wall=max(r['end'] for r in results)-min(r['begin'] for r in results)
                    row={'case':case,'repetition':repetition,'turn':turn,'wall_seconds':wall,
                         'aggregate_tps':sum(r['usage']['completion_tokens'] for r in results)/wall,
                         'overlap_seconds':max(0,min(r['last'] for r in results)-max(r['first'] for r in results)) if count==2 else None,
                         'requests':results}
                    if repetition >= 0:
                        out.write(json.dumps(row) + '\n')
                        out.flush()
                    print(case,repetition,turn,round(wall,3),round(row['aggregate_tps'],1),flush=True)
                    if case=='turns':
                        for lane,r in enumerate(results):
                            histories[lane].extend([r['assistant_message'], {'role':'user','content':f'Turn {turn+1}: give a different detailed example and explain it. Write at least 1000 words.'}])


if __name__ == '__main__':
    main()
