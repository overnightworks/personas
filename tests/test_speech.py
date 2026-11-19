from __future__ import annotations

import pytest

from personas.speech import SpeechRecord, speech_answered, speech_line_entered, speech_sent


def _spoken(record: SpeechRecord) -> SpeechRecord:
    return speech_answered(record, "One.", spoken=True)


def _written(record: SpeechRecord) -> SpeechRecord:
    return speech_answered(record, "One.", spoken=False)


@pytest.mark.parametrize(
    ("steps", "owed", "saying"),
    [
        pytest.param((), 0, None, id="nothing sent"),
        pytest.param((speech_line_entered,), 0, None, id="a line with no request and nothing owed is ignored"),
        pytest.param((speech_sent, _spoken), 1, "One.", id="a sentence answered as spoken owes its line"),
        pytest.param((speech_sent, _spoken, speech_line_entered), 0, "One.", id="the line brings the debt in"),
        pytest.param(
            (speech_sent, speech_line_entered, _spoken), 0, "One.", id="a line before its answer leaves nothing owed"
        ),
        pytest.param(
            (speech_sent, speech_line_entered, _written), 0, None, id="a line before a non-spoken answer is no debt"
        ),
        pytest.param((speech_sent, _written), 0, None, id="a sentence answered otherwise owes nothing"),
        pytest.param(
            (speech_sent, _spoken, speech_line_entered, speech_line_entered),
            0,
            "One.",
            id="a line owed to none is ignored",
        ),
        pytest.param((speech_sent, _spoken, speech_sent, _spoken), 2, "One.", id="two sentences owe two lines"),
    ],
)
def test_the_speech_record_counts_a_request_from_its_send_and_credits_his_line_only_while_one_is_open_or_owed(
    steps, owed, saying
):
    record = SpeechRecord()
    for step in steps:
        record = step(record)

    assert (record.owed, record.saying) == (owed, saying)
