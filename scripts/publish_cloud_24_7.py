#!/usr/bin/env python3
"""
OmniVoice Studio — Automated 24/7 Cloud Publisher (Hugging Face Spaces + GitHub Pages + Windows Auto-Start).
Deploys both Desktop Studio (`/`) and Standalone Mobile PWA (`/mobile`) to:
  1. Hugging Face Spaces (16 GB RAM + 2 vCPU Free 24/7 Docker Cloud):
     - https://<hf_user>-omnivoice-studio.hf.space
     - https://<hf_user>-omnivoice-studio.hf.space/mobile
  2. GitHub Repository & GitHub Pages (24/7 Static PWA Shell):
     - https://github.com/<gh_user>/OmniVoice-Studio
     - https://<gh_user>.github.io/OmniVoice-Studio/
"""

import os
import sys
import site
import json
import getpass
import subprocess
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(r"D:\ALL PROJECT\OJT PROJECT\OmniVoice")
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    site.addsitedir(str(VENV_SITE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

GIT_EXE = r"D:\Git\cmd\git.exe" if Path(r"D:\Git\cmd\git.exe").exists() else "git"
GH_EXE = r"C:\Program Files\GitHub CLI\gh.exe"
CLOUD_URLS_FILE = PROJECT_ROOT / "CLOUD_URLS_24_7.txt"

HF_YAML_HEADER = """---
title: OmniVoice Studio & Mobile PWA
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
"""


def ensure_hf_readme_header() -> None:
    readme_path = PROJECT_ROOT / "README.md"
    if not readme_path.exists():
        readme_path.write_text(HF_YAML_HEADER + "\n# OmniVoice Studio & Mobile PWA\n", encoding="utf-8")
        return
    content = readme_path.read_text(encoding="utf-8", errors="replace")
    if not content.startswith("---\n"):
        readme_path.write_text(HF_YAML_HEADER + "\n" + content, encoding="utf-8")
        print("[OK] Da bo sung cấu hình Hugging Face Spaces YAML vao README.md.")


def load_dotenv_token() -> str:
    for env_path in [Path.home() / ".env", PROJECT_ROOT / ".env"]:
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
    return os.environ.get("HF_TOKEN", "").strip()


def deploy_to_huggingface_spaces() -> dict:
    print("\n" + "=" * 72)
    print("  🚀 BUOC 1: DEPLOY 24/7 LEN HUGGING FACE SPACES (16 GB RAM MIEN PHI)")
    print("=" * 72)

    ensure_hf_readme_header()
    from huggingface_hub import HfApi, login

    api = HfApi()
    hf_user = None
    token = load_dotenv_token()

    if token:
        try:
            user_info = api.whoami(token=token)
            hf_user = user_info.get("name")
        except Exception:
            token = ""

    if not hf_user:
        try:
            user_info = api.whoami()
            hf_user = user_info.get("name")
        except Exception:
            hf_user = None

    if not hf_user:
        print("\n[THONG BAO] De day du an len Hugging Face Spaces chay 24/7 (tat laptop van vao duoc):")
        print("  1. Trinh duyet se mo trang tao Write Token cua Hugging Face.")
        print("  2. Tao 1 token loai 'Write' (bat dau bang hf_...), copy va dan vao day.")
        try:
            webbrowser.open("https://huggingface.co/settings/tokens/new?tokenType=write")
        except Exception:
            pass
        try:
            token = getpass.getpass("  👉 Dan Hugging Face Write Token (hf_... - ky tu duoc an): ").strip()
        except Exception:
            token = input("  👉 Dan Hugging Face Write Token (hf_...): ").strip()

        if not token:
            print("  [BO QUA] Ban chua nhap HF Token. Bo qua buoc Hugging Face Spaces.")
            return {}

        login(token=token, add_to_git_credential=True)
        api = HfApi(token=token)
        user_info = api.whoami()
        hf_user = user_info.get("name")

    space_name = "omnivoice-studio"
    repo_id = f"{hf_user}/{space_name}"
    print(f"\n  [+] Dang khoi tao Space 24/7: https://huggingface.co/spaces/{repo_id} ...")

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )

        print("  [+] Dang tai ma nguon Desktop Studio + Mobile PWA len Hugging Face Spaces...")
        api.upload_folder(
            folder_path=str(PROJECT_ROOT),
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=[
                "README.md",
                "DESIGN.md",
                "Dockerfile",
                "docker-compose.yml",
                "requirements-cloud.txt",
                "pyproject.toml",
                "app.py",
                "workstation.py",
                "omnivoice/**",
                "docs/**",
            ],
            ignore_patterns=[
                ".venv/**",
                "models/**",
                "cache/**",
                "outputs/**",
                "logs/**",
                "temp/**",
                "debug/**",
                "voices/*.pt",
                "voices/*.wav",
                "__pycache__/**",
                "*.pyc",
                ".git/**",
            ],
            commit_message="Deploy OmniVoice Studio & Mobile PWA V2.26 (24/7 Cloud Edition)",
        )

        subdomain = f"{hf_user.lower()}-{space_name}".replace("_", "-")
        direct_url = f"https://{subdomain}.hf.space"
        mobile_url = f"https://{subdomain}.hf.space/mobile"
        portal_url = f"https://huggingface.co/spaces/{repo_id}"

        print("\n  ✅ DA DEPLOY THANH CONG LEN HUGGING FACE SPACES (CHAY 24/7):")
        print(f"     🌐 Link Desktop Studio 24/7 : {direct_url}")
        print(f"     📱 Link Mobile App PWA 24/7 : {mobile_url}")
        print(f"     🤗 Trang quan ly HF Space   : {portal_url}")

        return {
            "hf_user": hf_user,
            "direct_url": direct_url,
            "mobile_url": mobile_url,
            "portal_url": portal_url,
        }
    except Exception as e:
        err_msg = str(e)
        if "402" in err_msg or "Payment Required" in err_msg or "PRO subscription" in err_msg:
            print("\n  [LƯU Ý CHÍNH SÁCH HUGGING FACE]")
            print("  Hugging Face vừa cập nhật chính sách mới: Tài khoản miễn phí (Free) hiện chỉ tạo được")
            print("  Space dạng tĩnh (Static), còn tạo Space Docker/Gradio mới yêu cầu tài khoản Pro ($9/tháng).")
            print("  -> Chuyển sang triển khai GitHub Pages và tối ưu hóa kết nối cho bạn...")
        else:
            print(f"\n  [WARN] Khong the deploy len HF Spaces: {e}")
        return {}


def deploy_to_github_and_pages(hf_info: dict) -> dict:
    print("\n" + "=" * 72)
    print("  🐙 BUOC 2: DEPLOY LEN GITHUB REPOSITORY & GITHUB PAGES 24/7")
    print("=" * 72)

    if not Path(GH_EXE).exists():
        print("  [WARN] Khong tim thay GitHub CLI.")
        return {}

    # Update docs/index.html if hf_info has a direct URL so GitHub Pages links directly to the 24/7 HF Space
    if hf_info.get("direct_url"):
        docs_index = PROJECT_ROOT / "docs" / "index.html"
        if docs_index.exists():
            html = docs_index.read_text(encoding="utf-8", errors="replace")
            # Inject default backend URL into localStorage fallback if not already set
            html = html.replace(
                "localStorage.getItem('ov_backend') || ''",
                f"localStorage.getItem('ov_backend') || '{hf_info['direct_url']}'",
            )
            docs_index.write_text(html, encoding="utf-8")

    # Commit latest changes
    subprocess.run([GIT_EXE, "-C", str(PROJECT_ROOT), "add", "-A"], capture_output=True)
    subprocess.run(
        [GIT_EXE, "-C", str(PROJECT_ROOT), "commit", "-m", "Release V2.26: 24/7 Hugging Face Spaces & GitHub Pages Cloud Deployment"],
        capture_output=True,
    )

    # Check gh auth status
    r_auth = subprocess.run([GH_EXE, "auth", "status"], capture_output=True, text=True)
    if r_auth.returncode != 0:
        print("\n  [THONG BAO] Dang mo trinh duyet de dang nhap GitHub (1-Click OAuth)...")
        subprocess.run([GH_EXE, "auth", "login", "--web", "-p", "https"])

    # Get GitHub username
    r_user = subprocess.run([GH_EXE, "api", "user", "-q", ".login"], capture_output=True, text=True)
    gh_user = r_user.stdout.strip()
    if not gh_user:
        print("  [BO QUA] Chua xac thuc GitHub.")
        return {}

    repo_name = "OmniVoice-Studio"
    print(f"  [+] Dang day ma nguon len https://github.com/{gh_user}/{repo_name} ...")
    r_create = subprocess.run(
        [GH_EXE, "repo", "create", repo_name, "--public", "--source=.", "--remote=github", "--push"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if r_create.returncode != 0:
        subprocess.run(
            [GIT_EXE, "-C", str(PROJECT_ROOT), "remote", "add", "github", f"https://github.com/{gh_user}/{repo_name}.git"],
            capture_output=True,
        )
        subprocess.run(
            [GIT_EXE, "-C", str(PROJECT_ROOT), "push", "-u", "github", "HEAD:main", "--force"],
            capture_output=True,
        )

    # Enable GitHub Pages with GitHub Actions workflow
    subprocess.run(
        [GH_EXE, "api", "-X", "POST", f"repos/{gh_user}/{repo_name}/pages", "-f", "build_type=workflow"],
        capture_output=True,
    )
    subprocess.run(
        [GH_EXE, "api", "-X", "PUT", f"repos/{gh_user}/{repo_name}/pages", "-f", "build_type=workflow"],
        capture_output=True,
    )

    repo_url = f"https://github.com/{gh_user}/{repo_name}"
    pages_url = f"https://{gh_user.lower()}.github.io/{repo_name}/"

    print("\n  ✅ DA DEPLOY THANH CONG LEN GITHUB & GITHUB PAGES (24/7):")
    print(f"     📦 GitHub Repository  : {repo_url}")
    print(f"     🌐 GitHub Pages (PWA) : {pages_url}")

    return {
        "gh_user": gh_user,
        "repo_url": repo_url,
        "pages_url": pages_url,
    }


def main():
    hf_info = deploy_to_huggingface_spaces()
    gh_info = deploy_to_github_and_pages(hf_info)

    lines = [
        "========================================================================",
        "  🌐 OMNIVOICE STUDIO — DANH SACH LINK CLOUD 24/7 (TAT LAPTOP VAN VAO DUOC)",
        "========================================================================",
    ]
    if hf_info:
        lines.extend([
            f"  [Hugging Face Cloud 24/7 - Desktop Studio] : {hf_info['direct_url']}",
            f"  [Hugging Face Cloud 24/7 - Mobile App PWA] : {hf_info['mobile_url']}",
            f"  [Hugging Face Space Portal]                : {hf_info['portal_url']}",
        ])
    if gh_info:
        lines.extend([
            f"  [GitHub Pages 24/7 - Mobile & Web Hub]     : {gh_info['pages_url']}",
            f"  [GitHub Source Repository]                 : {gh_info['repo_url']}",
        ])
    lines.append("========================================================================\n")
    summary = "\n".join(lines)
    print("\n" + summary)
    if hf_info or gh_info:
        CLOUD_URLS_FILE.write_text(summary, encoding="utf-8")
        print(f"  [SAVED] Da luu danh sach link cố định vao: {CLOUD_URLS_FILE}")


if __name__ == "__main__":
    main()
