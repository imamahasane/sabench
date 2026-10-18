from __future__ import annotations

from sabench.features.text import chat_participant_text


def test_chat_participant_text_keeps_participant_speech_only(tmp_path):
    cha = tmp_path / "S999.cha"
    cha.write_text(
        "@UTF8\n@Begin\n@ID:\teng|Pitt|PAR|70;|female|ProbableAD||Participant|20||\n"
        "*INV:\ttell me what you see . \x150_2000\x15\n"
        "*PAR:\tthere's a boy &uh <taking a> [//] taking cookies . \x152000_5000\x15\n"
        "%mor:\tpro:exist|there~cop|be&3S det:art|a n|boy\n"
        "*PAR:\tand I'm sayin(g) he's (be)cause the [/] the stool\n"
        "\tis falling . [+ gram] \x155000_9000\x15\n"
        "*INV:\tanything else ?\n@End\n",
        encoding="utf-8",
    )
    assert chat_participant_text(cha) == (
        "there's a boy uh taking a taking cookies "
        "and i'm saying he's because the the stool is falling"
    )


def test_collapse_repeats_trims_bursts_but_keeps_short_repetition():
    from sabench.features.text import collapse_repeats

    assert collapse_repeats("Okay. okay okay, okay okay okay the boy") == "Okay. okay okay, the boy"
    assert collapse_repeats("thank you thank you thank you thank you thank you bye") == "thank you thank you thank you bye"
    assert collapse_repeats("the the the stool is falling") == "the the the stool is falling"
    assert collapse_repeats("") == ""
