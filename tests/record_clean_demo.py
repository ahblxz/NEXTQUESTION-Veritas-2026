"""Record a competition-ready, caption-free NEXTQUESTION screen walkthrough.

Output: artifacts/hospital/NEXTQUESTION_full_voiceover_demo.webm
No captions, watermarks, voiceover, title slides or added text overlays.
Every frame shows the ACTUAL RUNNING interactive educational application.
All sample records displayed in the public-EHR explorer are de-identified
and taken from patient-disjoint validation cases, never final held-out tests.
"""
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright, expect
from app import CSS, THEME, demo


OUT = Path("artifacts/hospital")
PORT = 7884


def scroll_to(page, selector, seconds=2.8):
    """Smoothly pan the genuine interface into frame, without overlays."""
    locator = page.locator(selector).first
    expect(locator).to_be_visible(timeout=30000)
    locator.evaluate(
        """el => el.scrollIntoView({behavior:'smooth',block:'center'})"""
    )
    page.wait_for_timeout(int(seconds * 1000))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="127.0.0.1", server_port=PORT,
        prevent_thread_lock=True, share=False,
        theme=THEME, css=CSS, show_error=True
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, args=["--no-sandbox"]
            )
            context = browser.new_context(
                viewport={"width": 1600, "height": 900},
                record_video_dir=str(OUT / "_clean_recording"),
                record_video_size={"width": 1600, "height": 900},
                device_scale_factor=1,
            )
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(
                f"http://127.0.0.1:{PORT}/",
                wait_until="domcontentloaded", timeout=40000
            )
            expect(
                page.get_by_role("button", name="Start assessment")
            ).to_be_enabled(timeout=30000)

            # SECTION 1: what we built, the visible synthetic benchmark,
            # and the fictional input workflow (narration space).
            page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
            page.wait_for_timeout(4400)
            scroll_to(page, ".nq-benchmark", 3600 / 1000)
            scroll_to(page, "#nq-scenario", 2400 / 1000)

            # SECTION 2: start a complete synthetic assessment.
            page.get_by_role("button", name="Start assessment").click()
            expect(
                page.get_by_text("Possible explanations to investigate")
            ).to_be_visible(timeout=30000)
            scroll_to(page, ".nq-insights-column", 4200 / 1000)
            scroll_to(page, '.nq-card:has-text("Possible explanations")', 3400 / 1000)
            scroll_to(page, '.nq-card:has-text("Ranked information-gathering options")', 4000 / 1000)

            # Reveal an actual stored fictional test indicator (not text fakery).
            next_button = page.get_by_role(
                "button", name="Simulate suggested test result"
            )
            scroll_to(page, "#nq-acquire", 1500 / 1000)
            expect(next_button).to_be_enabled(timeout=30000)
            next_button.click()
            expect(page.get_by_text("1 of 16 test types")).to_be_visible(
                timeout=30000
            )
            scroll_to(page, '.nq-card:has-text("Sequential investigation history")', 4300 / 1000)
            scroll_to(page, '.nq-card:has-text("Possible explanations")', 3000 / 1000)

            # SECTION 3: manual fictional case and preexisting findings.
            scroll_to(page, "#nq-mode", 1300 / 1000)
            page.get_by_role("button", name="Reset").click()
            page.get_by_text("Manual fictional case", exact=True).click()
            page.get_by_role("button", name="Start assessment").click()
            manual = page.get_by_role(
                "button", name="Record fictional result"
            )
            expect(manual).to_be_enabled(timeout=30000)
            scroll_to(page, "#nq-manual-test", 2700 / 1000)
            manual.click()
            expect(page.get_by_text("1 of 16 test types")).to_be_visible(
                timeout=30000
            )
            scroll_to(page, '.nq-card:has-text("Sequential investigation history")', 3800 / 1000)

            # SECTION 4: true public de-identified hospital records.
            page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
            page.wait_for_timeout(2600)
            public_section = page.get_by_text(
                "Explore real public de-identified hospital records "
                "(separate research pilot)", exact=True
            )
            public_section.click()
            expect(
                page.get_by_role("button", name="Explore public encounter")
            ).to_be_enabled(timeout=30000)
            scroll_to(page, "#nq-real-case", 2700 / 1000)
            page.get_by_role("button", name="Explore public encounter").click()
            actual = page.get_by_role(
                "button", name="Reveal recorded lab finding"
            )
            expect(actual).to_be_enabled(timeout=30000)
            scroll_to(page, '.nq-card:has-text("Model-ranked discharge categories")', 4000 / 1000)
            scroll_to(page, '.nq-card:has-text("Ranked historical laboratory findings")', 2800 / 1000)
            actual.click()
            expect(
                page.get_by_text("Actual recorded lab flags").first
            ).to_be_visible(timeout=30000)
            scroll_to(page, '.nq-card:has-text("Actual recorded lab flags")', 3700 / 1000)
            scroll_to(page, '.nq-card:has-text("Model-ranked discharge categories")', 3600 / 1000)
            # A second actual recorded result updates the model again.
            expect(actual).to_be_enabled(timeout=30000)
            actual.click()
            scroll_to(page, '.nq-card:has-text("Actual recorded lab flags")', 3500 / 1000)
            scroll_to(page, '.nq-card:has-text("Ranked historical laboratory findings")', 3400 / 1000)
            scroll_to(page, ".nq-benchmark", 4400 / 1000)

            # End with the brand and explicit safety disclaimer already part
            # of the native UI. NO externally drawn caption or annotation.
            page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
            page.wait_for_timeout(4000)

            assert not errors, f"Browser JavaScript failures: {errors}"
            assert not page.locator("#nextquestion-video-caption").count(), (
                "No video caption overlay may be present"
            )
            video = page.video
            context.close()
            source = Path(video.path())
            dest = OUT / "NEXTQUESTION_full_voiceover_demo.webm"
            shutil.copyfile(source, dest)
            assert dest.stat().st_size > 100_000
            browser.close()
            print(
                "PASS: clean, caption-free actual-browser demo:",
                dest, dest.stat().st_size
            )
    finally:
        demo.close()


if __name__ == "__main__":
    main()
