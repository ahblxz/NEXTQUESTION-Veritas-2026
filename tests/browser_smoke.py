"""Live HTTP and phone-browser end-to-end test of the NEXTQUESTION hospital dashboard."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import THEME, CSS, SCENARIOS, demo, start, acquire, record_result, reset


def main():
    scenario = SCENARIOS[0]
    first = start("Synthetic demonstration", scenario["id"], scenario["complaint"], scenario["symptoms"], [])
    assert first[0]["cost"] == 0
    assert "Possible explanations" in first[2]
    second = acquire(first[0])
    assert second[0]["cost"] > 0
    assert len(second[0]["history"]) == 1
    manual = start("Manual fictional case", None, "fever and cough", [], [])
    assert record_result(manual[0], "Abnormal / flagged")[0]["cost"] > 0
    assert reset()[0] is None

    try:
        demo.launch(server_name="127.0.0.1", server_port=7870,
                    prevent_thread_lock=True, share=False, theme=THEME,
                    css=CSS, show_error=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:7870/", timeout=20) as response:
            assert response.status == 200
            assert len(response.read()) > 800
        with opener.open("http://127.0.0.1:7870/config", timeout=20) as response:
            config = json.loads(response.read())
        assert "NEXTQUESTION" in config.get("title", "")
        assert len(config["dependencies"]) >= 4

        from playwright.sync_api import sync_playwright, expect
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
            exceptions = []
            page.on("pageerror", lambda exc: exceptions.append(str(exc)))
            page.goto("http://127.0.0.1:7870/", wait_until="domcontentloaded", timeout=30000)
            # Wait for Gradio's client-side component mount before inspecting
            # DOM IDs (DOMContentLoaded alone is too early).
            expect(page.get_by_role("button", name="Start assessment")).to_be_visible(
                timeout=30000
            )
            Path("artifacts/hospital").mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path="artifacts/hospital/mobile_initial_polished.png", full_page=True
            )
            # Visual QA: no light default Gradio tiles in any intake form.
            # Playwright's locator pierces Gradio's shadow DOM; document.querySelector
            # alone doesn't, so inspect the actual computed styles on real inputs.
            pale = []
            for selector in (
                "#nq-mode", "#nq-scenario", "#nq-complaint",
                "#nq-symptoms", "#nq-flags"
            ):
                root = page.locator(selector)
                assert root.count() == 1, f"Missing functional control: {selector}"
                colors = root.evaluate("""(root) => {
                    const failures = [];
                    for (const el of [root, ...root.querySelectorAll('*')]) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width * rect.height < 550) continue;
                        const color = getComputedStyle(el).backgroundColor;
                        const nums = color.match(/[0-9.]+/g);
                        if (nums && nums.length >= 3 &&
                            +nums[0] > 193 && +nums[1] > 193 && +nums[2] > 193) {
                            failures.push(color + ' ' + el.tagName.toLowerCase());
                        }
                    }
                    return failures;
                }""")
                pale.extend([selector + ": " + value for value in colors])
            assert not pale, f"Pale/unstyled UI elements: {pale}"
            screen_width = page.evaluate("document.documentElement.scrollWidth")
            assert screen_width <= 396, (
                f"Horizontal overflow on 390px phone: {screen_width}"
            )
            print("PASS: dark form controls; no default white input boxes or horizontal phone overflow.")
            start_button = page.get_by_role("button", name="Start assessment")
            expect(start_button).to_be_enabled(timeout=30000)
            start_button.click()
            expect(page.get_by_text("Possible explanations to investigate")).to_be_visible(timeout=30000)
            next_button = page.get_by_role("button", name="Simulate suggested test result")
            expect(next_button).to_be_enabled(timeout=30000)
            next_button.click()
            expect(page.get_by_text("1 of 16 test types")).to_be_visible(timeout=30000)

            # Editing any input MUST immediately clear stale assessment state.
            page.get_by_label("Presenting complaint").fill(
                "headache and dizziness"
            )
            expect(next_button).to_be_disabled(timeout=30000)
            start_button.click()
            expect(next_button).to_be_enabled(timeout=30000)

            # Test the other user-facing workflow, not only the simulated auto-result.
            page.get_by_role("button", name="Reset").click()
            expect(next_button).to_be_disabled(timeout=30000)
            page.get_by_text("Manual fictional case", exact=True).click()
            start_button.click()
            manual_button = page.get_by_role("button", name="Record fictional result")
            expect(manual_button).to_be_enabled(timeout=30000)
            manual_button.click()
            expect(page.get_by_text("1 of 16 test types")).to_be_visible(timeout=30000)

            # Desktop smoke pass on a 1440px browser; same deployment serves both sizes.
            desktop = browser.new_page(viewport={"width": 1440, "height": 900})
            desktop.goto("http://127.0.0.1:7870/", wait_until="domcontentloaded", timeout=30000)
            expect(desktop.get_by_role("button", name="Start assessment")).to_be_enabled(timeout=30000)
            desktop.get_by_role("button", name="Start assessment").click()
            expect(desktop.get_by_text("Possible explanations to investigate")).to_be_visible(timeout=30000)
            desktop.screenshot(path="artifacts/hospital/desktop_demo_verified.png", full_page=True)
            desktop.close()
            assert not exceptions, f"Browser JS error: {exceptions}"
            Path("artifacts/hospital").mkdir(parents=True, exist_ok=True)
            page.screenshot(path="artifacts/hospital/mobile_demo_verified.png", full_page=True)
            browser.close()
        print("PASS: HTTP server, config, 8 callback outputs, synthetic + manual modes, "
              "phone viewport, reset, synthetic clicks, manual record click and updated dashboard.")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
