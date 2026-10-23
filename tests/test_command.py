"""Unit tests for the CLI (conversion itself is mocked)."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from pdf2md import cli
from pdf2md.pipeline import Result


def make_result(output: Path, failed=()) -> Result:
    return Result(
        markdown="md",
        page_text="txt",
        failed_pages=list(failed),
        total_pages=1,
        processed_pages=1,
        output_path=Path(output),
        work_dir=Path(str(output) + ".work"),
    )


class CommandTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdf = self.root / "a.pdf"
        self.pdf.write_bytes(b"%PDF-1.4")


class FileModeTests(CommandTestCase):
    def test_default_output_is_input_md(self):
        with mock.patch.object(cli, "_convert_one", return_value=make_result(self.pdf)) as convert:
            code = cli.main([str(self.pdf)])
        self.assertEqual(code, 0)
        self.assertEqual(convert.call_args.args[1], self.root / "a.md")

    def test_refuses_to_overwrite_the_input(self):
        code = cli.main([str(self.pdf), "-o", str(self.pdf)])
        self.assertEqual(code, 2)

    def test_failed_pages_exit_nonzero(self):
        with mock.patch.object(
            cli, "_convert_one", return_value=make_result(self.pdf, failed=[2])
        ), mock.patch("sys.stderr", io.StringIO()):
            code = cli.main([str(self.pdf)])
        self.assertEqual(code, 1)

    def test_dry_run_does_not_convert(self):
        buf = io.StringIO()
        with mock.patch.object(cli, "_convert_one") as convert, redirect_stdout(buf):
            code = cli.main([str(self.pdf), "--dry-run"])
        self.assertEqual(code, 0)
        convert.assert_not_called()
        self.assertIn("a.md", buf.getvalue())

    def test_missing_dependency_is_reported(self):
        stderr = io.StringIO()
        with mock.patch.object(
            cli, "_missing_dependencies", return_value=["rapidocr"]
        ), mock.patch("sys.stderr", stderr):
            code = cli.main([str(self.pdf)])
        self.assertEqual(code, 2)
        self.assertIn("rapidocr", stderr.getvalue())


class FolderModeTests(CommandTestCase):
    def make_tree(self) -> Path:
        src = self.root / "docs"
        (src / "sub").mkdir(parents=True)
        (src / "a.pdf").write_bytes(b"%PDF-1.4")
        (src / "sub" / "b.pdf").write_bytes(b"%PDF-1.4")
        return src

    def test_dry_run_accepts_a_relative_folder(self):
        self.make_tree()
        old_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli.main(["docs", "--dry-run"])
        finally:
            os.chdir(old_cwd)

        self.assertEqual(code, 0)
        self.assertIn("sub", buf.getvalue())
        self.assertIn("a.md", buf.getvalue())

    def test_output_preserves_source_layout(self):
        src = self.make_tree()
        dst = (self.root / "out").resolve()
        with mock.patch.object(
            cli, "_convert_one", side_effect=lambda pdf, output, args: make_result(output)
        ) as convert:
            code = cli.main([str(src), "-o", str(dst)])

        self.assertEqual(code, 0)
        outputs = sorted(call.args[1].relative_to(dst).as_posix() for call in convert.call_args_list)
        self.assertEqual(outputs, ["a.md", "sub/b.md"])

    def test_one_bad_file_does_not_stop_the_batch(self):
        src = self.make_tree()
        results = [RuntimeError("bad"), make_result(self.root / "b.md")]
        with mock.patch.object(cli, "_convert_one", side_effect=results) as convert:
            code = cli.main([str(src), "-o", str(self.root / "out")])

        self.assertEqual(convert.call_count, 2)
        self.assertEqual(code, 1)

    def test_default_output_folder_is_markdown_output(self):
        src = self.make_tree()
        with mock.patch.object(
            cli, "_convert_one", side_effect=lambda pdf, output, args: make_result(output)
        ) as convert:
            cli.main([str(src)])

        outputs = [call.args[1] for call in convert.call_args_list]
        self.assertTrue(all("markdown_output" in str(path) for path in outputs))


class EncryptionTests(CommandTestCase):
    def write_encrypted(self, password: str) -> Path:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt(password)
        with self.pdf.open("wb") as handle:
            writer.write(handle)
        return self.pdf

    def test_password_is_required(self):
        self.write_encrypted("secret")
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            code = cli.main([str(self.pdf)])
        self.assertEqual(code, 1)
        self.assertIn("password-protected", stderr.getvalue())

    def test_wrong_password_is_not_echoed(self):
        self.write_encrypted("secret")
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            code = cli.main([str(self.pdf), "--password", "hunter2"])
        self.assertEqual(code, 1)
        self.assertIn("did not unlock", stderr.getvalue())
        self.assertNotIn("hunter2", stderr.getvalue())

    def test_decryption_uses_a_deleted_temporary_copy(self):
        self.write_encrypted("secret")
        source, tmp = cli._readable_source(self.pdf, "secret")
        self.assertNotEqual(source, self.pdf)
        self.assertTrue(tmp.exists())
        tmp.unlink(missing_ok=True)
        self.assertFalse(tmp.exists())

    def test_pipeline_gets_the_original_as_fingerprint_source(self):
        self.write_encrypted("secret")
        args = cli.build_parser().parse_args([str(self.pdf), "--password", "secret"])
        captured = {}

        def fake_convert(pdf, output, **kwargs):
            captured["pdf"] = Path(pdf)
            captured["source_path"] = kwargs["source_path"]
            return make_result(output)

        with mock.patch.object(cli, "convert_pdf", side_effect=fake_convert):
            cli._convert_one(self.pdf, self.root / "x.md", args)

        self.assertNotEqual(captured["pdf"], self.pdf)
        self.assertFalse(captured["pdf"].exists())  # temp copy cleaned up
        self.assertEqual(captured["source_path"], self.pdf)


if __name__ == "__main__":
    unittest.main()
