from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_harness import RecordingHooks
from agent_harness.core.message import Message, ToolCall, ToolResult
from agent_harness.hooks import CompositeHooks, DefaultHooks
from agent_harness.llm.call_context import AgentIdentity, ModelCallContext
from agent_harness.llm.types import LLMResponse, Usage
from agent_harness.tool.base import ToolSchema


def context(name='main', seq=1):
    return ModelCallContext(AgentIdentity(name, None if name=='main' else 'main', 'main'), seq, 'agent', 'main')


async def test_complete_native_records_and_concurrent_agents(tmp_path):
    recorder = RecordingHooks(tmp_path / 'records.jsonl')
    hooks = CompositeHooks(DefaultHooks(), recorder)
    tool = ToolSchema(name='read', description='read', parameters={'type': 'object'})
    call = ToolCall(id='same-id', name='read', arguments={'path': '文.txt'})
    response = LLMResponse(message=Message.assistant(None, tool_calls=[call],
                            provider_metadata={'openai': {'reasoning_content': 'reason'}}),
                           usage=Usage(total_tokens=3), raw_response=object())
    async def write(name):
        for seq in (1, 2):
            await hooks.on_llm_response(name, [Message.user('read')], response,
                                         call_context=context(name, seq), tools=[tool])
            await hooks.on_tool_result(name, ToolResult(tool_call_id='same-id',
                name='read', content=f'{name}-{seq}', is_error=seq==2))
    await asyncio.gather(write('main'), write('child'))
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert len(records) == 8 and recorder.error is None
    for name in ('main', 'child'):
        own = [r for r in records if r['agent_name'] == name]
        assert [r['type'] for r in own] == ['llm_call', 'tool_result'] * 2
        assert [r['call']['seq'] for r in own if r['type']=='llm_call'] == [1, 2]
        assert own[0]['request']['tools'] == [tool.model_dump(mode='json')]
        assert own[0]['response']['message']['provider_metadata']['openai']['reasoning_content'] == 'reason'
        assert 'raw_response' not in own[0]['response']
        assert own[1]['result']['content'] == f'{name}-1'
    with pytest.raises(FileExistsError):
        RecordingHooks(recorder.path)
    assert len(recorder.path.read_text().splitlines()) == 8


@pytest.mark.parametrize('failure', ['serialize', 'append', 'close'])
async def test_write_failure_stops_only_recording(tmp_path, caplog, failure):
    recorder = RecordingHooks(tmp_path / 'trace.jsonl')
    real_open = Path.open
    class FailFile:
        def __enter__(self):
            return self
        def write(self, text):
            if failure == 'append':
                raise OSError('append failure')
        def __exit__(self, *args):
            if failure == 'close':
                raise OSError('close failure')
    def open_file(path, *args, **kwargs):
        return FailFile() if path == recorder.path else real_open(path, *args, **kwargs)
    result = ToolResult(tool_call_id='t', name='tool', content='ok')
    with patch.object(Path, 'open', open_file), patch('agent_harness.hooks.recording.logger.error') as log:
        await recorder.on_tool_result('main', object() if failure=='serialize' else result)
        first = recorder.error
        await recorder.on_tool_result('main', result)
    assert first is recorder.error and first is not None
    log.assert_called_once()


async def test_cancellation_waiting_for_lock_propagates(tmp_path):
    recorder = RecordingHooks(tmp_path / 'trace.jsonl')
    await recorder._lock.acquire()
    task = asyncio.create_task(recorder.on_tool_result('main', ToolResult(tool_call_id='t', name='x', content='ok')))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    recorder._lock.release()
    assert recorder.error is None and recorder.path.read_text() == ''
