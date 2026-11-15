"""Record a short, captioned, genuine browser walkthrough of the live web demo.

All cases are fictional and all results synthetic. Output: artifacts/hospital/
NEXTQUESTION_demo_raw.webm. This is a SILENT research screen recording, not
a narrated, finalized competition video.
"""
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright, expect
from app import CSS, THEME, demo


def caption(page, message):
    page.evaluate(
        """(message) => {
          let el = document.getElementById('nextquestion-video-caption');
          if (!el) {
            el = document.createElement('div');
            el.id = 'nextquestion-video-caption';
            el.style.cssText = 'position:fixed;bottom:12px;left:50%;transform:translateX(-50%);'
              + 'z-index:99999;background:rgba(0,0,0,.94);color:white;'
              + 'font:600 17px Arial,sans-serif;line-height:1.4;padding:10px 19px;'
              + 'border:1px solid #eeeeee;border-radius:10px;max-width:95vw;'
              + 'text-align:center;pointer-events:none';
            document.body.appendChild(el);
          }
          el.textContent = message;
        }""",
        message,
    )


def main():
    outdir = Path("artifacts/hospital")
    outdir.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7873,
                prevent_thread_lock=True, share=False,
                theme=THEME, css=CSS, show_error=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=str(outdir / "_raw_demo"),
                record_video_size={"width": 1280, "height": 720},
                device_scale_factor=1
            )
            page = context.new_page()
            page.goto("http://127.0.0.1:7873/", wait_until="domcontentloaded",
                      timeout=35000)
            expect(page.get_by_role("button", name="Start assessment")).to_be_enabled(timeout=30000)
            caption(page, "NEXTQUESTION | AI-guided information acquisition · RESEARCH SIMULATION")
            page.wait_for_timeout(4200)
            caption(page, "A fictional patient presents with symptoms · The AI considers several conditions")
            page.get_by_role("button", name="Start assessment").click()
            expect(page.get_by_text("Possible explanations to investigate")).to_be_visible(timeout=30000)
            page.wait_for_timeout(5500)
            caption(page, "The next test is ranked by expected information gain versus artificial effort")
            page.locator("text=Ranked information-gathering options").first.scroll_into_view_if_needed()
            page.wait_for_timeout(4000)
            caption(page, "Acquire a fictional result: posterior and test order update")
            button = page.get_by_role("button", name="Simulate suggested test result")
            expect(button).to_be_enabled(timeout=30000)
            button.click()
            expect(page.get_by_text("1 of 16 test types")).to_be_visible(timeout=30000)
            page.wait_for_timeout(6500)
            caption(page, "Record an already known fictional finding in manual case mode")
            page.get_by_role("button", name="Reset").click()
            page.get_by_text("Manual fictional case", exact=True).click()
            page.get_by_role("button", name="Start assessment").click()
            manual = page.get_by_role("button", name="Record fictional result")
            expect(manual).to_be_enabled(timeout=30000)
            manual.click()
            expect(page.get_by_text("1 of 16 test types")).to_be_visible(timeout=30000)
            page.wait_for_timeout(6500)
            caption(page, "Held-out synthetic study: 2.79 tests adaptive vs 3.53 fixed; 7.31 effort pts each")
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(6500)
            caption(page, "Urgent-class false negatives remain: 74 / 328 · NOT medical advice")
            page.wait_for_timeout(4200)
            caption(page, "Not medically validated. No proven cost savings. This is a research prototype.")
            page.wait_for_timeout(4000)
            video = page.video
            context.close()
            source = Path(video.path())
            dest = outdir / "NEXTQUESTION_demo_raw.webm"
            shutil.copyfile(source, dest)
            assert dest.stat().st_size > 10000, dest
            browser.close()
            print("PASS: recorded genuine fictional-case walkthrough:", dest, dest.stat().st_size)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
