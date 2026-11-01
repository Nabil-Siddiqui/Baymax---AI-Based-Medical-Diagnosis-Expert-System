/* =====================================================================
   Baymax - Frontend Script
   =====================================================================
   This file powers the interactive parts of the Baymax website:

   - Dropdown-driven symptom selection with removable chips.
   - Live syncing of selected symptoms into a hidden form field.
   - Enabling/disabling the "Diagnose" button.
   - Clear-all action.
   - Rotating Baymax greetings in the speech bubble.

   The script is intentionally framework-free and safe to include on
   pages that may not contain every element it looks for. Each module
   below is gated by `if (...)` checks so it gracefully no-ops when
   the corresponding HTML structure isn't on the page.
   ===================================================================== */

(function () {
    "use strict";

    /* -----------------------------------------------------------------
       Utility helpers
       ----------------------------------------------------------------- */

    /**
     * Find a single element by attribute hook.
     */
    function $(selector, root) {
        return (root || document).querySelector(selector);
    }

    /**
     * Find all elements by attribute hook.
     */
    function 
$$
(selector, root) {
        return Array.prototype.slice.call(
            (root || document).querySelectorAll(selector)
        );
    }

    /**
     * Read inline JSON from a <script type="application/json"> tag.
     * Returns null if the tag is missing or the JSON is malformed.
     */
    function readInlineJSON(elementId) {
        const node = document.getElementById(elementId);
        if (!node) return null;
        try {
            return JSON.parse(node.textContent || "null");
        } catch (err) {
            console.warn("Baymax: failed to parse inline JSON", elementId, err);
            return null;
        }
    }

    /* -----------------------------------------------------------------
       Symptom selector module
       -----------------------------------------------------------------
       Active only when the homepage form is present.
       ----------------------------------------------------------------- */

    function initSymptomSelector() {
        const form          = $("[data-baymax-form]");
        const select        = $("[data-baymax-symptom-select]");
        const addButton     = $("[data-baymax-add-symptom]");
        const chipList      = $("[data-baymax-chip-list]");
        const chipEmpty     = $("[data-baymax-chip-empty]");
        const csvInput      = $("[data-baymax-symptoms-csv]");
        const diagnoseBtn   = $("[data-baymax-diagnose-button]");
        const clearAllBtn   = $("[data-baymax-clear-symptoms]");

        // If the page doesn't have the symptom form, do nothing.
        if (!form || !select || !addButton || !chipList || !csvInput || !diagnoseBtn) {
            return;
        }

        // Build a lookup table of symptom IDs -> labels from the
        // inline JSON shipped by the server. Falls back to the
        // dropdown's own option text if JSON is missing.
        const symptomData = readInlineJSON("baymax-symptoms-data") || [];
        const labelById = {};
        symptomData.forEach(function (entry) {
            if (entry && entry.id) {
                labelById[entry.id] = entry.label || entry.id;
            }
        });

        // Track currently selected symptoms in insertion order.
        const selected = [];

        function getLabel(id) {
            if (labelById[id]) return labelById[id];
            // Fallback: pull the option text from the dropdown.
            const opt = select.querySelector('option[value="' + cssEscape(id) + '"]');
            return opt ? opt.textContent.trim() : id;
        }

        /**
         * Lightweight CSS.escape polyfill suitable for our IDs
         * (which only use [a-z0-9_]).
         */
        function cssEscape(value) {
            return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
        }

        /**
         * Synchronize the hidden CSV field, the empty-state text, and
         * the diagnose button's enabled state to match `selected`.
         */
        function syncFormState() {
            csvInput.value = selected.join(",");

            if (selected.length === 0) {
                if (chipEmpty) chipEmpty.style.display = "";
                diagnoseBtn.disabled = true;
            } else {
                if (chipEmpty) chipEmpty.style.display = "none";
                diagnoseBtn.disabled = false;
            }
        }

        /**
         * Render the chip list from the current `selected` array.
         * We rebuild the list each time for simplicity and correctness.
         */
        function renderChips() {
            // Clear existing chips but keep the empty-state element.
            const existingChips =
$$
(".chip", chipList);
            existingChips.forEach(function (chip) { chip.remove(); });

            selected.forEach(function (symptomId) {
                const label = getLabel(symptomId);

                const chip = document.createElement("span");
                chip.className = "chip";
                chip.dataset.symptomId = symptomId;

                const text = document.createElement("span");
                text.textContent = label;
                chip.appendChild(text);

                const removeBtn = document.createElement("button");
                removeBtn.type = "button";
                removeBtn.className = "chip__remove";
                removeBtn.setAttribute("aria-label", "Remove " + label);
                removeBtn.textContent = "×";
                removeBtn.addEventListener("click", function () {
                    removeSymptom(symptomId);
                });

                chip.appendChild(removeBtn);
                chipList.appendChild(chip);
            });
        }

        function addSymptom(symptomId) {
            if (!symptomId) return;
            if (selected.indexOf(symptomId) !== -1) {
                // Already added; gently flash the existing chip.
                const existing = chipList.querySelector(
                    '[data-symptom-id="' + cssEscape(symptomId) + '"]'
                );
                if (existing) {
                    existing.classList.remove("chip--flash");
                    // Force reflow so the animation can replay.
                    void existing.offsetWidth;
                    existing.classList.add("chip--flash");
                }
                return;
            }

            selected.push(symptomId);
            renderChips();
            syncFormState();
        }

        function removeSymptom(symptomId) {
            const idx = selected.indexOf(symptomId);
            if (idx === -1) return;
            selected.splice(idx, 1);
            renderChips();
            syncFormState();
        }

        function clearAll() {
            if (selected.length === 0) return;
            selected.length = 0;
            renderChips();
            syncFormState();
            // Reset the dropdown back to its placeholder.
            select.selectedIndex = 0;
            select.focus();
        }

        // Initial render in case the page was reloaded with a state.
        renderChips();
        syncFormState();

        // "+ Add" click: take whatever the dropdown currently has and add it.
        addButton.addEventListener("click", function () {
            const value = select.value;
            if (!value) return;
            addSymptom(value);
            // Reset dropdown and refocus for fast multi-add.
            select.selectedIndex = 0;
            select.focus();
        });

        // Pressing Enter inside the dropdown should add the current selection.
        select.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                addButton.click();
            }
        });

        // Clear all button.
        if (clearAllBtn) {
            clearAllBtn.addEventListener("click", function () {
                clearAll();
            });
        }

        // Defensive: prevent submitting an empty form.
        form.addEventListener("submit", function (event) {
            if (selected.length === 0) {
                event.preventDefault();
                diagnoseBtn.disabled = true;
            }
        });
    }

    /* -----------------------------------------------------------------
       Greeting rotator module
       -----------------------------------------------------------------
       Active on any page that has a Baymax speech bubble.
       ----------------------------------------------------------------- */

    const GREETINGS = [
        "Hello, I am Baymax, your personal healthcare companion.",
        "On a scale of one to ten, how would you rate your symptoms?",
        "I am here to help. Tell me what you are feeling.",
        "Take your time. Select every symptom that applies.",
        "I cannot diagnose without your input. Please share your symptoms.",
        "Your wellbeing matters. Let us begin."
    ];

    function initGreetingRotator() {
        const bubble  = $("[data-baymax-bubble]");
        const target  = $("[data-baymax-greeting]");

        if (!bubble || !target) return;

        // Respect users who prefer reduced motion: don't rotate, just
        // show the first greeting and stop.
        const reducedMotion =
            window.matchMedia &&
            window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        if (reducedMotion) return;

        let index = 0;

        function showNextGreeting() {
            index = (index + 1) % GREETINGS.length;
            const next = GREETINGS[index];

            // Cross-fade by quickly toggling opacity through inline style.
            target.style.transition = "opacity 280ms ease";
            target.style.opacity = "0";

            window.setTimeout(function () {
                target.textContent = next;
                target.style.opacity = "1";
            }, 280);
        }

        // Rotate every 5.5 seconds. Long enough to read comfortably.
        window.setInterval(showNextGreeting, 5500);
    }

    /* -----------------------------------------------------------------
       Boot
       ----------------------------------------------------------------- */

    function boot() {
        try {
            initSymptomSelector();
        } catch (err) {
            console.error("Baymax: symptom selector failed to initialize", err);
        }

        try {
            initGreetingRotator();
        } catch (err) {
            console.error("Baymax: greeting rotator failed to initialize", err);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        // Document is already parsed (e.g. script loaded at the bottom).
        boot();
    }
})();