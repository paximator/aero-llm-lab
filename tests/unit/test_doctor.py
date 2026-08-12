from aerollm import doctor


def test_core_doctor_reports_success(capsys) -> None:
    assert doctor.main([]) == 0
    assert "status=ok profile=core" in capsys.readouterr().out


def test_version_reports_missing_package() -> None:
    assert doctor._version("aerollm-package-that-does-not-exist") == "missing"
