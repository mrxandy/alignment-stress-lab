from alignment_stress_lab.baseline import prompt_rows


def test_prompt_rows_preserve_source_line_numbers_and_text() -> None:
    assert prompt_rows([" first  \n", "\n", "第二条\r\n", "  \n"]) == [
        (1, " first  "),
        (3, "第二条"),
    ]
