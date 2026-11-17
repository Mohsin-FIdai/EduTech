/**
 * Frontend Client JavaScript for Pakistani English & Roman-Urdu Lecture & Meeting Intelligence System
 * Complete Modern SaaS / EdTech Theme (media_1788990330717.jpg)
 */

document.addEventListener("DOMContentLoaded", () => {
    // Top-Level DOM Elements
    const recordingDashboard = document.getElementById("recordingDashboard");
    const uploadsSection = document.getElementById("uploadsSection");
    const topNavBackBtn = document.getElementById("topNavBackBtn");
    const topNavBreadcrumb = document.getElementById("topNavBreadcrumb");
    const alertBox = document.getElementById("alertBox");
    const globalSearchInput = document.getElementById("globalSearchInput");

    // Sidebar Navigation Buttons
    const navDashboard = document.getElementById("navDashboard");
    const navUploads = document.getElementById("navUploads");
    const navNotes = document.getElementById("navNotes");
    const navQuizzes = document.getElementById("navQuizzes");
    const navExports = document.getElementById("navExports");
    const navSettings = document.getElementById("navSettings");

    // Media & Player Elements
    const audioPlayer = document.getElementById("audioPlayer");
    const audioVisualizerBlock = document.getElementById("audioVisualizerBlock");
    const waveformWrapper = document.getElementById("waveformWrapper");
    const waveformSvg = document.getElementById("waveformSvg");
    const waveformProgressFill = document.getElementById("waveformProgressFill");
    const waveformPinsOverlay = document.getElementById("waveformPinsOverlay");
    const currentTimeDisplay = document.getElementById("currentTimeDisplay");
    const totalDurationDisplay = document.getElementById("totalDurationDisplay");

    // Upload & Progress Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const progressContainer = document.getElementById("progressContainer");
    const progressBarFill = document.getElementById("progressBarFill");
    const progressPercent = document.getElementById("progressPercent");
    const progressLabel = document.getElementById("progressLabel");
    const recordingsList = document.getElementById("recordingsList");
    const refreshBtn = document.getElementById("refreshBtn");

    // Context & Staged File Selection Elements
    const stagedFileCard = document.getElementById("stagedFileCard");
    const stagedFileName = document.getElementById("stagedFileName");
    const stagedFileMeta = document.getElementById("stagedFileMeta");
    const stagedFileIcon = document.getElementById("stagedFileIcon");
    const removeStagedFileBtn = document.getElementById("removeStagedFileBtn");
    const submitProcessBtn = document.getElementById("submitProcessBtn");
    const uploadTopic = document.getElementById("uploadTopic");
    const uploadRecordingType = document.getElementById("uploadRecordingType");

    // Stepper Items
    const stepItems = {
        upload: document.getElementById("step-upload"),
        preprocess: document.getElementById("step-preprocess"),
        transcribe: document.getElementById("step-transcribe"),
        normalize: document.getElementById("step-normalize"),
        intelligence: document.getElementById("step-intelligence"),
        ready: document.getElementById("step-ready")
    };

    // State Variables
    let activeStageTimer = null;
    let stepperRunMaxPct = 0;
    let stepperRunMaxIdx = -1;
    let isPipelineBusy = false;

    function clearActiveStageTimer() {
        if (activeStageTimer !== null) {
            clearInterval(activeStageTimer);
            activeStageTimer = null;
        }
    }
    function startActiveStageTimer(tickFn, intervalMs = 2000) {
        clearActiveStageTimer();
        activeStageTimer = setInterval(tickFn, intervalMs);
        return activeStageTimer;
    }
    function resetStepperRun() {
        clearActiveStageTimer();
        stepperRunMaxPct = 0;
        stepperRunMaxIdx = -1;
    }
    let currentRecording = null;
    let currentIntelligence = null;
    let currentSegments = [];
    let currentTranscriptMode = "side-by-side"; // 'side-by-side', 'original', 'normalized'
    let currentQuizIdx = 0;
    let quizQuestions = [];
    let lastDashboardRecordingId = null;
    let stagedFile = null;

    // =========================================================================
    // =========================================================================
    // Markdown + Built-in LaTeX Math Rendering Utility (Zero External CDN Needed)
    // =========================================================================
    function formatLatexFallbackHtml(texExpr) {
        let s = (texExpr || "").trim();
        const escapeHtml = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        s = escapeHtml(s);

        // Strip formatting commands and spacing escapes
        s = s
            .replace(/\\(?:,|;|:|quad|qquad|!|space)\b/g, " ")
            .replace(/\\\s+/g, " ")
            .replace(/\\(?:text|mathrm|mathbf|mathit|textit|mathsf|operatorname|boldsymbol)\{([^{}]*)\}/g, "$1")
            .replace(/\\overline\{([^{}]*)\}/g, '<span style="text-decoration:overline;">$1</span>')
            .replace(/\\left([(\[{|.])/g, (_, ch) => (ch === "." ? "" : ch))
            .replace(/\\right([)\]}|.])/g, (_, ch) => (ch === "." ? "" : ch))
            .replace(/\\(max|min|log|ln|exp|avg|mod|sin|cos|tan)\b/g, "$1");

        // Convert Greek letters, operators, arrows, relations BEFORE \frac so nested _{i=1}^{n} inside numerators expand first
        s = s
            .replace(/\\sum(?![a-zA-Z])/g, "∑").replace(/\\prod(?![a-zA-Z])/g, "∏").replace(/\\int(?![a-zA-Z])/g, "∫")
            .replace(/\\times\b/g, "×").replace(/\\cdot\b/g, "·").replace(/\\div\b/g, "÷")
            .replace(/\\pm\b/g, "±").replace(/\\mp\b/g, "∓")
            .replace(/\\approx\b/g, "≈").replace(/\\sim\b/g, "∼").replace(/\\equiv\b/g, "≡")
            .replace(/\\leq\b|\\le\b/g, "≤").replace(/\\geq\b|\\ge\b/g, "≥").replace(/\\neq\b|\\ne\b/g, "≠")
            .replace(/\\rightarrow\b|\\to\b/g, "→").replace(/\\leftarrow\b/g, "←")
            .replace(/\\Rightarrow\b|\\implies\b/g, "⇒").replace(/\\Leftrightarrow\b|\\iff\b/g, "⇔")
            .replace(/\\infty\b/g, "∞").replace(/\\propto\b/g, "∝")
            .replace(/\\forall\b/g, "∀").replace(/\\exists\b/g, "∃")
            .replace(/\\in\b/g, "∈").replace(/\\notin\b/g, "∉")
            .replace(/\\subset\b/g, "⊂").replace(/\\cup\b/g, "∪").replace(/\\cap\b/g, "∩")
            .replace(/\\therefore\b/g, "∴").replace(/\\because\b/g, "∵")
            .replace(/\\(?:dots|ldots|cdots)\b/g, "…")
            .replace(/\\alpha\b/g, "α").replace(/\\beta\b/g, "β").replace(/\\gamma\b/g, "γ")
            .replace(/\\delta\b/g, "δ").replace(/\\Delta\b/g, "Δ").replace(/\\epsilon\b/g, "ε")
            .replace(/\\lambda\b/g, "λ").replace(/\\mu\b/g, "μ").replace(/\\sigma\b/g, "σ")
            .replace(/\\Sigma\b/g, "Σ").replace(/\\theta\b/g, "θ").replace(/\\pi\b/g, "π")
            .replace(/\\omega\b/g, "ω").replace(/\\Omega\b/g, "Ω");

        // Subscripts & superscripts cleanly formatted with HTML tags
        s = s
            .replace(/_\{([^{}]+)\}/g, '<sub style="font-size:0.76em;">$1</sub>')
            .replace(/_([A-Za-z0-9]+)/g, '<sub style="font-size:0.76em;">$1</sub>')
            .replace(/\^\{([^{}]+)\}/g, '<sup style="font-size:0.76em;">$1</sup>')
            .replace(/\^([A-Za-z0-9]+)/g, '<sup style="font-size:0.76em;">$1</sup>');

        // Handle stacked fractions & square roots after inner braces are expanded
        for (let pass = 0; pass < 2; pass++) {
            s = s.replace(/\\sqrt\{([^{}]*)\}/g, "√($1)");
            s = s.replace(/\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}/g,
                '<span style="display:inline-flex; flex-direction:column; align-items:center; vertical-align:middle; margin:0 4px; font-size:0.92em; line-height:1.15;"><span style="border-bottom:1.5px solid #334155; padding:0 5px 2px;">$1</span><span style="padding:2px 5px 0;">$2</span></span>'
            );
        }

        // Clean any remaining LaTeX command names, braces, or stray backslashes
        s = s
            .replace(/\\([a-zA-Z]+)/g, "$1")
            .replace(/[{}]/g, "")
            .replace(/\\/g, "")
            .replace(/\s{2,}/g, " ")
            .trim();
        return s;
    }

    function renderMathHtml(texExpr, isDisplay) {
        const cleanTex = (texExpr || "").trim();
        if (!cleanTex) return "";
        if (window.katex && typeof window.katex.renderToString === "function") {
            try {
                const kHtml = window.katex.renderToString(cleanTex, {
                    displayMode: !!isDisplay,
                    throwOnError: true,
                    strict: false,
                });
                return isDisplay
                    ? `<div class="math-display-box">${kHtml}</div>`
                    : `<span class="math-inline">${kHtml}</span>`;
            } catch (_) {}
        }
        const prettyHtml = formatLatexFallbackHtml(cleanTex);
        return isDisplay
            ? `<div class="math-display-box" style="font-family:Georgia,'Times New Roman',serif; font-size:1.05rem; font-weight:500; color:#0F172A; letter-spacing:0.01em;">${prettyHtml}</div>`
            : `<span class="math-inline" style="font-family:Georgia,'Times New Roman',serif; font-weight:500;">${prettyHtml}</span>`;
    }

    function renderMarkdownToHtml(md) {
        if (!md) return "";
        let src = String(md).replace(/\\n(?![a-zA-Z])/g, "\n").replace(/\r\n/g, "\n");
        const tokens = [];
        const stash = (htmlSnippet) => {
            const key = `@@MD_TOK_${tokens.length}@@`;
            tokens.push(htmlSnippet);
            return key;
        };

        // 1. Protect fenced code blocks before escaping or math/italic rules
        src = src.replace(/```([\w-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
            const safe = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").trimEnd();
            return stash(`<pre style="background:#0F172A; color:#F8FAFC; padding:0.8rem 1rem; border-radius:10px; font-size:0.8rem; overflow-x:auto; margin:0.75rem 0; font-family:'JetBrains Mono',monospace; line-height:1.45;"><code>${safe}</code></pre>`);
        });

        // 2. Protect display math: $$ ... $$ and \[ ... \] (plus trailing unclosed block during live streaming)
        src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => stash(renderMathHtml(tex, true)));
        src = src.replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => stash(renderMathHtml(tex, true)));
        src = src.replace(/\$\$([\s\S]+)$/g, (_, tex) => stash(renderMathHtml(tex, true)));

        // 3. Protect inline code before inline math
        src = src.replace(/`([^`\n]+)`/g, (_, code) => {
            const safe = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            return stash(`<code style="background:#F1F5F9; color:#4338CA; padding:2px 6px; border-radius:5px; font-size:0.84em; font-family:'JetBrains Mono',monospace;">${safe}</code>`);
        });

        // 4. Protect inline math: $ ... $ (single-line, non-empty, not currency like $50) and \( ... \)
        src = src.replace(/\\\(([^\n]+?)\\\)/g, (_, tex) => stash(renderMathHtml(tex, false)));
        src = src.replace(/\$([^\n$]+?)\$/g, (m, tex) => {
            if (/^\d[\d,.\s]*$/.test(tex.trim())) return m;
            return stash(renderMathHtml(tex, false));
        });

        // 5. Escape raw HTML entities in remaining prose
        let html = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

        // 6. Tables: | ... |
        html = html.replace(/((?:\|[^\n]+\|\n?){2,})/g, (match) => {
            const rawRows = match.trim().split("\n").map(r => r.trim()).filter(Boolean);
            if (rawRows.length < 2) return match;
            let tableHtml = '<div style="overflow-x:auto; margin:0.85rem 0; border:1px solid #E2E8F0; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.04);"><table style="width:100%; border-collapse:collapse; font-size:0.83rem; background:#FFFFFF;">';
            let inThead = true;
            rawRows.forEach((row, i) => {
                if (/^\|?[\s:-]+\|/.test(row) && row.includes("---")) {
                    inThead = false;
                    return;
                }
                const cells = row.split("|").map(c => c.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
                if (inThead && i === 0) {
                    tableHtml += '<thead style="background:#F1F5F9; border-bottom:2px solid #CBD5E1;"><tr>';
                    cells.forEach(c => { tableHtml += `<th style="padding:9px 12px; text-align:left; font-weight:700; color:#1E293B;">${c}</th>`; });
                    tableHtml += "</tr></thead><tbody>";
                } else {
                    tableHtml += '<tr style="border-bottom:1px solid #F1F5F9;">';
                    cells.forEach(c => { tableHtml += `<td style="padding:8px 12px; color:#334155; line-height:1.5;">${c}</td>`; });
                    tableHtml += "</tr>";
                }
            });
            tableHtml += "</tbody></table></div>";
            return stash(tableHtml);
        });

        // 7. Block-level formatting line-by-line (headings, lists, blockquotes, paragraphs)
        const inlineFmt = (s) => s
            .replace(/\*\*(.+?)\*\*/g, '<strong style="color:#0F172A; font-weight:700;">$1</strong>')
            .replace(/(^|[^\w*])\*([^*\n]+?)\*([^\w*]|$)/g, '$1<em>$2</em>$3');

        const lines = html.split("\n");
        const out = [];
        let listTag = null;
        const closeList = () => {
            if (listTag) {
                out.push(`</${listTag}>`);
                listTag = null;
            }
        };

        for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line) {
                closeList();
                continue;
            }
            if (/^@@MD_TOK_\d+@@$/.test(line)) {
                closeList();
                out.push(line);
                continue;
            }
            let m;
            if ((m = line.match(/^#{4,6}\s+(.*)$/))) {
                closeList();
                out.push(`<h4>${inlineFmt(m[1])}</h4>`);
            } else if ((m = line.match(/^###\s+(.*)$/))) {
                closeList();
                out.push(`<h3>${inlineFmt(m[1])}</h3>`);
            } else if ((m = line.match(/^##\s+(.*)$/))) {
                closeList();
                out.push(`<h2>${inlineFmt(m[1])}</h2>`);
            } else if ((m = line.match(/^#\s+(.*)$/))) {
                closeList();
                out.push(`<h2>${inlineFmt(m[1])}</h2>`);
            } else if (/^---+$/.test(line)) {
                closeList();
                out.push('<hr style="border:none; border-top:1px solid #E2E8F0; margin:0.9rem 0;" />');
            } else if ((m = line.match(/^&gt;\s*(.*)$/))) {
                closeList();
                const qText = m[1];
                if (/^\*\*Key Principle:\*\*/i.test(qText)) {
                    out.push(`<div style="background:#F0FDF4; border-left:4px solid #10B981; border-radius:0 8px 8px 0; padding:0.65rem 0.9rem; margin:0.65rem 0; font-size:0.85rem; color:#166534;">${inlineFmt(qText)}</div>`);
                } else if (/^\*\*Exam (?:Key )?Takeaway:\*\*/i.test(qText)) {
                    out.push(`<div style="background:#EFF6FF; border-left:4px solid #3B82F6; border-radius:0 8px 8px 0; padding:0.65rem 0.9rem; margin:0.65rem 0; font-size:0.85rem; color:#1E40AF;">${inlineFmt(qText)}</div>`);
                } else {
                    out.push(`<div style="background:#F8FAFC; border-left:3px solid #6366F1; border-radius:0 8px 8px 0; padding:0.55rem 0.85rem; margin:0.55rem 0; font-size:0.84rem; color:#475569;">${inlineFmt(qText)}</div>`);
                }
            } else if ((m = line.match(/^(?:[•\-]|\*(?!\*))\s+(.*)$/))) {
                if (listTag !== "ul") {
                    closeList();
                    listTag = "ul";
                    out.push("<ul>");
                }
                out.push(`<li>${inlineFmt(m[1])}</li>`);
            } else if ((m = line.match(/^\d+[.)]\s+(.*)$/))) {
                if (listTag !== "ol") {
                    closeList();
                    listTag = "ol";
                    out.push("<ol>");
                }
                out.push(`<li>${inlineFmt(m[1])}</li>`);
            } else {
                closeList();
                // Clean stray single-asterisk note lines like "* Where ..." into clean notes
                const cleanedPara = line.replace(/^\*\s+(?!\*)/, "");
                out.push(`<p>${inlineFmt(cleanedPara)}</p>`);
            }
        }
        closeList();

        let rendered = out.join("\n");
        for (let i = 0; i < tokens.length; i++) {
            rendered = rendered.replace(`@@MD_TOK_${i}@@`, tokens[i]);
        }
        return rendered;
    }

    // Initialize
    initApp();

    async function initApp() {
        setupEventListeners();
        renderWaveformSvg();
        switchView("uploads");
        await loadRecordings(false); // Clean empty landing page on startup/refresh (no auto-open)
    }

    // =========================================================================
    // Event Listeners Setup
    // =========================================================================
    function setupEventListeners() {
        // Sidebar Navigation (Safe routing when no lecture is active)
        navDashboard.addEventListener("click", () => {
            if (currentRecording) {
                switchView("dashboard");
                switchWorkspaceTab("overview");
            } else {
                switchView("uploads");
                showAlert("Please select a recording from the library below or upload a new file to open the workspace.", "info");
            }
        });
        navUploads.addEventListener("click", () => switchView("uploads"));
        navNotes.addEventListener("click", () => {
            if (currentRecording) {
                switchView("dashboard");
                switchWorkspaceTab("notes");
            } else {
                switchView("uploads");
                showAlert("Please select a recording from the library below or upload a new file to view Study Notes.", "info");
            }
        });
        navQuizzes.addEventListener("click", () => {
            if (currentRecording) {
                switchView("dashboard");
                switchWorkspaceTab("quiz");
            } else {
                switchView("uploads");
                showAlert("Please select a recording from the library below or upload a new file to view Practice Quizzes.", "info");
            }
        });
        navExports.addEventListener("click", () => {
            if (currentRecording) {
                openExportModal();
            } else {
                showAlert("Please select a recording from the library below or upload a new file to export deliverables.", "info");
            }
        });
        navSettings.addEventListener("click", () => {
            showAlert("Settings: Running on Local FastAPI + Whisper Core v2.0 (Roman Urdu / English Bilingual ASR)", "info");
        });

        // Switch / New Lecture Button (Lecture Hero Header)
        document.getElementById("switchLectureBtn")?.addEventListener("click", () => {
            switchView("uploads");
        });

        // Workspace Navigation Tabs
        document.querySelectorAll(".workspace-tab-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const tabKey = btn.getAttribute("data-tab");
                if (tabKey) switchWorkspaceTab(tabKey);
            });
        });

        // Overview Section Quick Navigation Links
        document.getElementById("overviewJumpAssignmentsBtn")?.addEventListener("click", () => {
            switchWorkspaceTab("assignments");
        });

        // Top Nav Back Button
        topNavBackBtn.addEventListener("click", () => {
            if (uploadsSection.classList.contains("hidden")) {
                switchView("uploads");
            } else if (currentRecording) {
                switchView("dashboard");
            }
        });

        // Library Real-time Search Input
        const librarySearchInput = document.getElementById("librarySearchInput");
        if (librarySearchInput) {
            librarySearchInput.addEventListener("input", (e) => {
                librarySearchTerm = e.target.value.toLowerCase().trim();
                applyLibraryFilters();
            });
        }

        // Library Category Filter Chips
        document.querySelectorAll(".filter-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                document.querySelectorAll(".filter-chip").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                activeLibraryFilter = chip.getAttribute("data-filter") || "all";
                applyLibraryFilters();
            });
        });

        // Drag & Drop File Upload
        if (dropZone && fileInput) {
            ["dragenter", "dragover"].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.add("dragover");
                }, false);
            });

            ["dragleave", "drop"].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.remove("dragover");
                }, false);
            });

            dropZone.addEventListener("drop", (e) => {
                const files = e.dataTransfer.files;
                if (files.length > 0) stageFile(files[0]);
            });

            fileInput.addEventListener("change", (e) => {
                if (e.target.files.length > 0) stageFile(e.target.files[0]);
            });
        }

        // Staged File Actions & Enter Submission
        removeStagedFileBtn?.addEventListener("click", clearStagedFile);
        submitProcessBtn?.addEventListener("click", triggerStagedProcessing);
        uploadRecordingType?.addEventListener("change", updateSubmitBtnLabel);
        uploadTopic?.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                triggerStagedProcessing();
            }
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && stagedFile && uploadsSection && !uploadsSection.classList.contains("hidden")) {
                const tag = (e.target && e.target.tagName) ? e.target.tagName : "";
                const id = (e.target && e.target.id) ? e.target.id : "";
                if (id === "globalSearchInput" || ["BUTTON", "A", "TEXTAREA", "SELECT"].includes(tag)) return;
                e.preventDefault();
                triggerStagedProcessing();
            }
        });

        // Refresh Button
        refreshBtn.addEventListener("click", () => loadRecordings(false));

        // Media Player Time Sync & Waveform Scrubbing
        audioPlayer?.addEventListener("timeupdate", onMediaTimeUpdate);
        audioPlayer?.addEventListener("loadedmetadata", () => {
            if (totalDurationDisplay) totalDurationDisplay.innerText = formatDurationSimple(currentRecording?.duration_seconds || audioPlayer.duration || 0);
        });

        if (waveformWrapper) {
            waveformWrapper.addEventListener("click", (e) => {
                const rect = waveformWrapper.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const ratio = Math.max(0, Math.min(1, clickX / rect.width));
                const targetTime = ratio * (currentRecording?.duration_seconds || audioPlayer?.duration || 1);
                seekMedia(targetTime);
            });
        }

        // Transcript vs Timeline Tabs
        const btnTabTranscript = document.getElementById("btnTabTranscript");
        const btnTabTimeline = document.getElementById("btnTabTimeline");
        const transcriptSubFilters = document.getElementById("transcriptSubFilters");
        const transcriptSegmentsList = document.getElementById("transcriptSegmentsList");
        const timelineFeedList = document.getElementById("timelineFeedList");

        btnTabTranscript?.addEventListener("click", () => {
            btnTabTranscript.classList.add("active");
            btnTabTimeline?.classList.remove("active");
            transcriptSubFilters?.classList.remove("hidden");
            transcriptSegmentsList?.classList.remove("hidden");
            timelineFeedList?.classList.add("hidden");
        });

        btnTabTimeline?.addEventListener("click", () => {
            btnTabTimeline.classList.add("active");
            btnTabTranscript?.classList.remove("active");
            transcriptSubFilters?.classList.add("hidden");
            transcriptSegmentsList?.classList.add("hidden");
            timelineFeedList?.classList.remove("hidden");
            renderTimelineFeed();
        });

        // Transcript Mode Filter Buttons
        document.querySelectorAll(".mode-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                currentTranscriptMode = btn.getAttribute("data-mode");
                renderTranscripts();
            });
        });

        // Transcript Text Filter
        const trFilter = document.getElementById("transcriptFilterInput");
        if (trFilter) {
            trFilter.addEventListener("input", (e) => {
                renderTranscripts(e.target.value.toLowerCase().trim());
            });
        }

        // Global Search
        if (globalSearchInput) {
            globalSearchInput.addEventListener("input", (e) => {
                const q = e.target.value.toLowerCase().trim();
                if (trFilter) trFilter.value = q;
                renderTranscripts(q);
            });
        }

        // Bottom Card Tabs: Practice Quiz vs Video Highlights vs Exports
        const tabPracticeQuiz = document.getElementById("tabPracticeQuiz");
        const tabHighlights = document.getElementById("tabHighlights");
        const tabExportsBottom = document.getElementById("tabExportsBottom");
        const interactiveQuizWrapper = document.getElementById("interactiveQuizWrapper");
        const highlightsWrapper = document.getElementById("highlightsWrapper");
        const quizCounterDisplay = document.getElementById("quizCounterDisplay");

        tabPracticeQuiz?.addEventListener("click", () => {
            tabPracticeQuiz.classList.add("active");
            tabHighlights?.classList.remove("active");
            interactiveQuizWrapper?.classList.remove("hidden");
            highlightsWrapper?.classList.add("hidden");
            if (quizCounterDisplay) quizCounterDisplay.style.display = "block";
        });

        tabHighlights?.addEventListener("click", () => {
            tabHighlights.classList.add("active");
            tabPracticeQuiz?.classList.remove("active");
            interactiveQuizWrapper?.classList.add("hidden");
            highlightsWrapper?.classList.remove("hidden");
            if (quizCounterDisplay) quizCounterDisplay.style.display = "none";
            renderHighlightsFeed();
        });

        tabExportsBottom?.addEventListener("click", () => {
            openExportModal();
        });

        // Quiz Navigation Buttons
        document.getElementById("quizPrevBtn")?.addEventListener("click", () => {
            if (currentQuizIdx > 0) {
                currentQuizIdx--;
                renderActiveQuizQuestion();
            }
        });

        document.getElementById("quizNextBtn")?.addEventListener("click", () => {
            if (currentQuizIdx < quizQuestions.length - 1) {
                currentQuizIdx++;
                renderActiveQuizQuestion();
            }
        });

        document.getElementById("quizRevealBtn")?.addEventListener("click", () => {
            revealCurrentQuizAnswer();
        });

        document.getElementById("quizRegenerateBtn")?.addEventListener("click", (e) => {
            triggerQuizRegeneration(e.currentTarget);
        });

        // AI Q&A Form
        const qaForm = document.getElementById("qaForm");
        if (qaForm) {
            qaForm.addEventListener("submit", (e) => {
                e.preventDefault();
                const query = document.getElementById("qaInput").value.trim();
                if (query) submitQuestion(query);
            });
        }

        // AI Q&A Suggestions (Event Delegation) & Refresh Button
        const qaSectionEl = document.getElementById("qaSection");
        if (qaSectionEl) {
            qaSectionEl.addEventListener("click", (e) => {
                const btn = e.target.closest(".qa-suggestion-btn");
                if (btn) {
                    const q = btn.getAttribute("data-q");
                    if (q) {
                        const inputEl = document.getElementById("qaInput");
                        if (inputEl) inputEl.value = q;
                        submitQuestion(q);
                    }
                }
            });
        }

        document.getElementById("qaRefreshBtn")?.addEventListener("click", () => {
            refreshQAChat(currentRecording, currentIntelligence);
            showAlert("Chatbot conversation refreshed.", "info");
        });

        // Export Modal & Buttons
        document.getElementById("exportAllBtn")?.addEventListener("click", openExportModal);
        document.getElementById("exportPdfBtn")?.addEventListener("click", () => {
            if (currentRecording) window.open(`/api/recordings/${currentRecording.id}/export?content_type=full&format=pdf`, "_blank");
        });
        document.getElementById("downloadNotesPdfBtn")?.addEventListener("click", () => {
            if (currentRecording) {
                window.open(`/api/recordings/${currentRecording.id}/export?content_type=notes&format=pdf`, "_blank");
            } else {
                showAlert("Please select a recording to download study notes.", "info");
            }
        });
        document.getElementById("closeExportModalBtn")?.addEventListener("click", closeExportModal);
        document.getElementById("modalExportPdf")?.addEventListener("click", () => triggerExport("pdf"));
        document.getElementById("modalExportNotesPdf")?.addEventListener("click", () => {
            if (currentRecording) {
                window.open(`/api/recordings/${currentRecording.id}/export?content_type=notes&format=pdf`, "_blank");
                closeExportModal();
            }
        });
        document.getElementById("modalExportDocx")?.addEventListener("click", () => triggerExport("docx"));
        document.getElementById("modalExportMd")?.addEventListener("click", () => triggerExport("md"));
        document.getElementById("modalExportTxt")?.addEventListener("click", () => triggerExport("txt"));

                // Reprocess Modal Functions & State
        let reprocessTargetId = null;

        function openReprocessModal(recordingId, filename = "") {
            reprocessTargetId = recordingId;
            const modal = document.getElementById("reprocessModal");
            const sub = document.getElementById("reprocessModalSubtitle");
            if (sub) {
                sub.innerText = filename ? `Choose how you want to re-process "${filename}".` : "Choose how you want to re-process this recording.";
            }
            if (modal) modal.classList.remove("hidden");
        }
        window.openReprocessModal = openReprocessModal;

        function closeReprocessModal() {
            const modal = document.getElementById("reprocessModal");
            if (modal) modal.classList.add("hidden");
            reprocessTargetId = null;
        }

        async function reprocessIntelligenceOnly(recordingId) {
            if (isPipelineBusy) { showAlert("Processing is already in progress. Please wait for the current stage to complete.", "info"); return; }
            isPipelineBusy = true;
            resetStepperRun();
            hideAlert();
            switchView("uploads");
            try {
                updateStepper("intelligence", 60, "Re-analyzing lecture: Extracting summaries, notes, assignments & quizzes...");
                let reprocessElapsedSec = 0;
                const reprocessTimer = startActiveStageTimer(() => {
                    reprocessElapsedSec += 2;
                    const mins = Math.floor(reprocessElapsedSec / 60);
                    const secs = reprocessElapsedSec % 60;
                    const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                    const p = Math.min(88, 60 + Math.floor(reprocessElapsedSec / 15));
                    let phase = "Extracting topics & key concepts";
                    if (reprocessElapsedSec > 90) phase = "Generating summaries & conceptual quizzes";
                    updateStepper("intelligence", p, `Re-analyzing lecture: ${phase} (${timeStr} elapsed, running on CUDA GPU)...`);
                }, 2000);

                let intelRes;
                try {
                    intelRes = await fetch(`/api/recordings/${recordingId}/intelligence/generate`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" }
                    });
                } finally {
                    clearActiveStageTimer();
                }

                if (!intelRes.ok) {
                    const err = await intelRes.json().catch(() => ({}));
                    throw new Error(err.detail || `Intelligence generation failed (HTTP ${intelRes.status})`);
                }

                updateStepper("ready", 90, "Re-generating dense vector embeddings for semantic search & Q&A...");
                try {
                    await fetch(`/api/recordings/${recordingId}/embeddings/generate`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" }
                    });
                } catch (e) {
                    console.warn("Embeddings generation warning:", e);
                }

                updateStepper("ready", 100, "AI Re-analysis complete!");
                isPipelineBusy = false; clearActiveStageTimer(); showAlert("Lecture intelligence re-extracted successfully! Loading updated workspace...", "success");

                await loadRecordings(false);

                setTimeout(() => {
                    progressContainer.classList.add("hidden");
                    openDashboard(recordingId);
                }, 600);

            } catch (err) {
                progressContainer.classList.add("hidden");
                isPipelineBusy = false; clearActiveStageTimer(); showAlert(`<strong>Re-analysis Failed:</strong> ${err.message}`, "error");
                await loadRecordings(false);
            }
        }

        // Reprocess Modal Listeners
        document.getElementById("closeReprocessModalBtn")?.addEventListener("click", closeReprocessModal);
        document.getElementById("cancelReprocessModalBtn")?.addEventListener("click", closeReprocessModal);

        document.getElementById("reprocessFastOptionBtn")?.addEventListener("click", () => {
            const targetId = reprocessTargetId || currentRecording?.id;
            closeReprocessModal();
            if (targetId) {
                reprocessIntelligenceOnly(targetId);
            }
        });

        document.getElementById("reprocessFullOptionBtn")?.addEventListener("click", () => {
            const targetId = reprocessTargetId || currentRecording?.id;
            closeReprocessModal();
            if (targetId) {
                startProcessing(targetId, true);
            }
        });

        // Details Modal Listeners
        document.getElementById("closeDetailsModalBtn")?.addEventListener("click", closeDetailsModal);
        document.getElementById("viewFullSummaryBtn")?.addEventListener("click", () => {
            const detailed = currentIntelligence?.summary_detailed || currentIntelligence?.summary_short || "No detailed summary available.";
            const renderedHtml = renderMarkdownToHtml(detailed);
            openDetailsModal("Comprehensive Executive Briefing", `<div style="font-size:0.9rem; line-height:1.6; color:#1E293B;">${renderedHtml}</div>`);
        });

        window.openMasterStudyGuideModal = function() {
            const notes = currentIntelligence?.lecture_notes || {};
            const guideMd = notes.markdown_study_guide || "";
            const studyTopics = notes.study_topics || [];
            const overview = notes.lecture_overview || "";
            const defs = notes.definitions || [];
            const concepts = notes.key_concepts || [];

            let downloadBanner = "";
            if (currentRecording) {
                downloadBanner = `
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.25rem; padding:0.8rem 1.1rem; background:linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%); border-radius:10px; border:1px solid #C7D2FE; flex-wrap:wrap; gap:0.6rem;">
                        <div>
                            <strong style="font-size:0.88rem; color:#1E1B4B; display:block;">High-Yield Lecture Study Guide</strong>
                            <span style="font-size:0.76rem; color:#4338CA;">Complete pedagogical breakdown, formulas, and definitions.</span>
                        </div>
                        <button onclick="window.open('/api/recordings/${currentRecording.id}/export?content_type=notes&format=pdf', '_blank')" class="btn btn-primary btn-sm" style="display:inline-flex; align-items:center; gap:6px; font-weight:700; box-shadow:0 2px 6px rgba(79, 70, 229, 0.25);">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                            <span>Download Notes (PDF)</span>
                        </button>
                    </div>
                `;
            }

            let content = downloadBanner;

            // Fallback status indicator
            if (notes.is_fallback || notes.analyzer_used === "local_extractive_fallback") {
                content += `
                    <div style="background:#FEF3C7; border:1px solid #FDE68A; border-radius:8px; padding:0.6rem 0.9rem; margin-bottom:1rem; font-size:0.8rem; color:#92400E; display:flex; align-items:center; gap:8px;">
                        <span style="font-weight:700;">Extractive Fallback Mode:</span>
                        <span>${notes.fallback_reason || 'Local LLM unavailable; notes extracted using deterministic semantic fallback.'}</span>
                    </div>
                `;
            }

            // If Master Study Guide is available, render formatted textbook guide
            if (guideMd) {
                content += `
                    <div style="background:#EEF2FF; border-left:4px solid #4F46E5; border-radius:0 8px 8px 0; padding:0.75rem 1rem; margin-bottom:1.25rem; font-size:0.85rem; color:#312E81;">
                        <strong>High-Yield Master Study Guide:</strong> Complete pedagogical review with formulas, comparison tables, worked examples, and exam Q&As.
                    </div>
                    <div class="master-study-guide-body" style="font-size:0.88rem; line-height:1.6; color:#1E293B;">
                        ${renderMarkdownToHtml(guideMd)}
                    </div>
                `;
            } else if (studyTopics.length > 0) {
                // Topic cards view
                if (overview) {
                    content += `<div style="background:#EEF2FF; border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.87rem; color:#312E81; line-height:1.5;">${overview}</div>`;
                }
                content += `<h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.75rem; color:#0F172A;">Topic-Wise Study Guide</h4>`;
                studyTopics.forEach((t, idx) => {
                    const spans = t.spans || [];
                    const startSec = spans.length > 0 ? spans[0].start : (t.start_time || 0);
                    const endSec = spans.length > 0 ? spans[spans.length - 1].end : (t.end_time || 0);
                    const examTag = t.exam_relevant ? `<span style="background:#FEF3C7; color:#92400E; font-size:0.7rem; padding:1px 6px; border-radius:8px; font-weight:600; margin-left:0.4rem;">EXAM</span>` : "";
                    const kps = (t.key_points || []).slice(0, 6).map(kp => `<li style="font-size:0.83rem; margin-bottom:0.25rem;">${kp}</li>`).join("");
                    const ex = (t.examples || []).length > 0 ? `<div style="font-size:0.8rem; color:#475569; margin-top:0.4rem;"><em>Example: ${t.examples[0]}</em></div>` : "";
                    const fms = (t.formulas || []).map(f => `<span style="background:#F1F5F9; color:#0F172A; font-size:0.75rem; padding:2px 6px; border-radius:6px; font-family:monospace; margin-right:4px;">${f}</span>`).join("");
                    content += `
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.75rem;">
                            <div style="display:flex; align-items:center; margin-bottom:0.4rem; flex-wrap:wrap; gap:0.4rem;">
                                <strong style="color:#4F46E5; font-size:0.9rem;">${idx+1}. ${t.title}</strong>${examTag}
                                <button onclick="seekMedia(${startSec}); document.getElementById('detailsModal')?.style && (document.getElementById('detailsModal').style.display='none');" style="background:#EEF2FF; color:#4F46E5; border:none; border-radius:8px; font-size:0.72rem; padding:2px 8px; cursor:pointer;">${formatDurationSimple(startSec)}–${formatDurationSimple(endSec)}</button>
                            </div>
                            <p style="font-size:0.83rem; color:#334155; margin-bottom:0.5rem; line-height:1.5;">${t.explanation || ""}</p>
                            ${kps ? `<ul style="padding-left:1.2rem; margin:0 0 0.4rem 0;">${kps}</ul>` : ""}
                            ${ex}
                            ${fms ? `<div style="margin-top:0.4rem;">${fms}</div>` : ""}
                        </div>
                    `;
                });
            } else if (defs.length > 0 || concepts.length > 0) {
                content += `<h4 style="font-size:0.95rem; font-weight:700; margin-bottom:0.5rem;">Key Definitions</h4>`;
                if (defs.length === 0) content += `<p style="color:#64748B; font-size:0.85rem;">No explicit definitions logged.</p>`;
                defs.forEach(d => {
                    content += `<div style="margin-bottom:0.6rem;"><strong style="color:#4F46E5;">${d.term}</strong>: <span style="font-size:0.85rem;">${d.definition}</span></div>`;
                });
                content += `<h4 style="font-size:0.95rem; font-weight:700; margin:1rem 0 0.5rem 0;">Key Conceptual Themes</h4><div style="display:flex; flex-wrap:wrap; gap:0.4rem;">`;
                concepts.forEach(c => { content += `<span class="topic-tag-chip">${c}</span>`; });
                content += `</div>`;
            } else {
                content += `<p style="color:#64748B;">No study notes available. Regenerate the intelligence report to analyse this lecture.</p>`;
            }

            openDetailsModal("High-Yield Master Study Guide", content);
        };
        document.getElementById("viewAllNotesBtn")?.addEventListener("click", window.openMasterStudyGuideModal);


        document.getElementById("viewAllAssignmentsBtn")?.addEventListener("click", () => {
            const asgns = currentIntelligence?.assignments || [];
            let content = "";
            if (asgns.length === 0) content = `<p style="color:#64748B;">No assignments recorded for this session.</p>`;
            asgns.forEach(a => {
                content += `
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:1rem; margin-bottom:0.75rem;">
                        <h4 style="font-weight:700; color:#0F172A; margin-bottom:0.25rem;">${a.topic || 'Assignment'}</h4>
                        <p style="font-size:0.82rem; color:#475569; margin-bottom:0.5rem;">${a.required_tasks || a.description}</p>
                        <div style="font-size:0.75rem; display:flex; gap:0.5rem;">
                            <span class="due-chip">Due: ${a.deadline || 'Specified in audio'}</span>
                            <span class="submission-portal-pill">Portal: ${a.submission_method || 'LMS Portal'}</span>
                        </div>
                    </div>
                `;
            });
            openDetailsModal("Assignments & Course Deliverables", content);
        });

        document.getElementById("viewAllActionItemsBtn")?.addEventListener("click", () => {
            const items = currentIntelligence?.action_items || [];
            let content = "";
            if (items.length === 0) content = `<p style="color:#64748B;">No action items detected.</p>`;
            items.forEach((item, idx) => {
                content += `<div style="padding:0.5rem 0; border-bottom:1px solid #E2E8F0;"><input type="checkbox" id="m-act-${idx}"> <label for="m-act-${idx}"><strong>${item.task}</strong> ${item.deadline ? `<span style="color:#64748B;">(${item.deadline})</span>` : ''}</label></div>`;
            });
            openDetailsModal("Action Items & Meeting Deliverables", content);
        });

        // Share Button
        document.getElementById("shareBtn")?.addEventListener("click", () => {
            navigator.clipboard?.writeText(window.location.href);
            showAlert("Lecture workspace link copied to clipboard!", "success");
        });
    }

    // =========================================================================
    // =========================================================================
    // Workspace Tab Switching
    // =========================================================================
    let activeWorkspaceTab = "overview";

    function switchWorkspaceTab(tabKey) {
        activeWorkspaceTab = tabKey;
        const tabMap = {
            "overview": { btn: "tabWorkspaceOverview", section: "workspaceSectionOverview", label: "Overview" },
            "transcript": { btn: "tabWorkspaceTranscript", section: "workspaceSectionTranscript", label: "Transcript" },
            "notes": { btn: "tabWorkspaceNotes", section: "workspaceSectionNotes", label: "Study Notes" },
            "assignments": { btn: "tabWorkspaceAssignments", section: "workspaceSectionAssignments", label: "Assignments & Actions" },
            "quiz": { btn: "tabWorkspaceQuiz", section: "workspaceSectionQuiz", label: "Practice Quiz" },
            "qa": { btn: "tabWorkspaceQA", section: "workspaceSectionQA", label: "AI Assistant" }
        };

        // Update active tab buttons
        document.querySelectorAll(".workspace-tab-btn").forEach(btn => btn.classList.remove("active"));
        const activeEntry = tabMap[tabKey];
        if (activeEntry) {
            document.getElementById(activeEntry.btn)?.classList.add("active");
        }

        // Hide all sections, show active section
        document.querySelectorAll(".workspace-section").forEach(sec => {
            sec.classList.add("hidden");
            sec.classList.remove("active");
        });

        if (activeEntry) {
            const targetSec = document.getElementById(activeEntry.section);
            if (targetSec) {
                targetSec.classList.remove("hidden");
                targetSec.classList.add("active");
            }
            if (topNavBreadcrumb && uploadsSection.classList.contains("hidden")) {
                const lectureTitle = currentRecording?.filename ? currentRecording.filename.replace(/\.[^/.]+$/, "") : "Workspace";
                topNavBreadcrumb.innerText = `${lectureTitle} • ${activeEntry.label}`;
            }
        }
    }
    window.switchWorkspaceTab = switchWorkspaceTab;

    // View Switching
    // =========================================================================
    function switchView(viewName) {
        hideAlert();
        navDashboard.classList.toggle("active", viewName === "dashboard");
        navUploads.classList.toggle("active", viewName === "uploads");

        if (viewName === "dashboard") {
            recordingDashboard.classList.remove("hidden");
            uploadsSection.classList.add("hidden");
            const lectureTitle = currentRecording?.filename ? currentRecording.filename.replace(/\.[^/.]+$/, "") : "Workspace";
            const tabNames = {
                "overview": "Overview",
                "transcript": "Transcript",
                "notes": "Study Notes",
                "assignments": "Assignments & Actions",
                "quiz": "Practice Quiz",
                "qa": "AI Assistant"
            };
            topNavBreadcrumb.innerText = `${lectureTitle} • ${tabNames[activeWorkspaceTab] || 'Overview'}`;
        } else if (viewName === "uploads") {
            recordingDashboard.classList.add("hidden");
            uploadsSection.classList.remove("hidden");
            topNavBreadcrumb.innerText = "Workspace — Select or Upload";
        }
    }

    // =========================================================================
    // Alerts & Utilities
    // =========================================================================
    function showAlert(message, type = "error") {
        alertBox.className = `alert-box alert-${type}`;
        const iconSvg = type === "error"
            ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
            : (type === "success"
                ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`
                : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`);
        alertBox.innerHTML = `<span>${iconSvg}</span> <div>${message}</div>`;
        alertBox.classList.remove("hidden");
    }

    function hideAlert() {
        alertBox.classList.add("hidden");
        alertBox.innerHTML = "";
    }

    function formatBytes(bytes, decimals = 1) {
        if (!bytes || bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function formatDurationSimple(seconds) {
        if (isNaN(seconds) || seconds < 0) return "00:00";
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    function formatDurationExtended(seconds) {
        if (isNaN(seconds) || seconds < 0) return "00:00:00";
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hrs > 0) {
            return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    function extractStartSeconds(tsStr) {
        if (tsStr === undefined || tsStr === null || tsStr === "") return 0;
        if (typeof tsStr === "number") return Math.max(0, tsStr);
        const sStr = String(tsStr).trim();

        // 1. Match [HH:MM:SS] or HH:MM:SS
        const hhmmssMatch = sStr.match(/\[?(\d+):(\d+):(\d+(?:\.\d+)?)/);
        if (hhmmssMatch) {
            return parseInt(hhmmssMatch[1], 10) * 3600 + parseInt(hhmmssMatch[2], 10) * 60 + parseFloat(hhmmssMatch[3]);
        }

        // 2. Match [MM:SS] or MM:SS (e.g. [05:16 - 10:37] or 05:16)
        const mmssMatch = sStr.match(/\[?(\d+):(\d+(?:\.\d+)?)/);
        if (mmssMatch) {
            return parseInt(mmssMatch[1], 10) * 60 + parseFloat(mmssMatch[2]);
        }

        // 3. Match [12.5s] or 12.50s or 12.5
        const secMatch = sStr.match(/\[?([\d\.]+)\s*s?/);
        return secMatch ? parseFloat(secMatch[1]) : 0;
    }

    // =========================================================================
    // Waveform & Chapter Pins Visualizer
    // =========================================================================
    function renderWaveformSvg() {
        if (!waveformSvg) return;
        const barCount = 120;
        let paths = "";
        for (let i = 0; i < barCount; i++) {
            const x = (i / barCount) * 1000;
            const height = 12 + Math.abs(Math.sin(i * 0.18) * 35) + (Math.random() * 8);
            const y = 30 - (height / 2);
            paths += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="4" height="${height.toFixed(1)}" rx="2" fill="#818CF8" opacity="0.75" />`;
        }
        waveformSvg.innerHTML = paths;
    }

    function renderChapterPins(duration, notes) {
        if (!waveformPinsOverlay) return;
        waveformPinsOverlay.innerHTML = "";
        if (!duration || duration <= 0) return;

        const studyTopics = (notes && notes.study_topics) || [];
        const legacyTopics = (notes && notes.topics) || [];
        const pinsData = [];

        if (studyTopics.length > 0) {
            studyTopics.forEach((t, idx) => {
                const sTime = t.start_time !== undefined ? t.start_time : (t.spans && t.spans[0] ? t.spans[0].start : 0);
                pinsData.push({
                    time: sTime,
                    label: t.title || `Topic ${idx + 1}`
                });
            });
        } else if (legacyTopics.length > 0) {
            legacyTopics.forEach((t, idx) => {
                const fraction = (idx + 1) / (legacyTopics.length + 1);
                pinsData.push({
                    time: fraction * duration,
                    label: t
                });
            });
        } else {
            pinsData.push({ time: duration * 0.2, label: "Introduction & Foundations" });
            pinsData.push({ time: duration * 0.5, label: "Core Concepts & Analysis" });
            pinsData.push({ time: duration * 0.8, label: "Assignments & Wrap Up" });
        }

        pinsData.forEach(p => {
            const percent = Math.min(99, Math.max(1, (p.time / duration) * 100));
            const pinEl = document.createElement("div");
            pinEl.className = "chapter-pin";
            pinEl.style.left = `${percent}%`;
            pinEl.innerHTML = `<span class="chapter-pin-tooltip">${formatDurationSimple(p.time)} • ${p.label}</span>`;
            pinEl.addEventListener("click", (e) => {
                e.stopPropagation();
                seekMedia(p.time);
            });
            waveformPinsOverlay.appendChild(pinEl);
        });
    }

    function onMediaTimeUpdate() {
        const player = audioPlayer;
        if (!player) return;
        const current = player.currentTime || 0;
        const total = currentRecording?.duration_seconds || player.duration || 1;

        if (currentTimeDisplay) currentTimeDisplay.innerText = formatDurationSimple(current);
        if (totalDurationDisplay) totalDurationDisplay.innerText = formatDurationSimple(total);

        const pct = Math.min(100, Math.max(0, (current / total) * 100));
        if (waveformProgressFill) {
            waveformProgressFill.style.width = `${pct}%`;
        }

        highlightActiveTranscript(current);
    }

    function highlightActiveTranscript(currentSec) {
        const rows = document.querySelectorAll(".transcript-turn");
        rows.forEach(r => {
            const start = parseFloat(r.getAttribute("data-start") || 0);
            const end = parseFloat(r.getAttribute("data-end") || 0);
            if (currentSec >= start && currentSec <= end) {
                if (!r.classList.contains("active-speaking")) {
                    document.querySelectorAll(".transcript-turn.active-speaking").forEach(el => el.classList.remove("active-speaking"));
                    r.classList.add("active-speaking");
                }
            }
        });
    }

    let pendingSeekHandler = null;

    window.seekMedia = function(secondsOrTs) {
        let sec = 0;
        if (typeof secondsOrTs === "number") {
            sec = secondsOrTs;
        } else if (typeof secondsOrTs === "string") {
            sec = extractStartSeconds(secondsOrTs);
        }
        if (isNaN(sec) || !isFinite(sec)) sec = 0;
        sec = Math.max(0, Math.min(sec, currentRecording?.duration_seconds || 86400));

        if (!audioPlayer) return;

        if (pendingSeekHandler) {
            audioPlayer.removeEventListener("loadedmetadata", pendingSeekHandler);
            pendingSeekHandler = null;
        }

        const applySeekAndPlay = () => {
            const maxDur = (audioPlayer.duration && isFinite(audioPlayer.duration))
                ? audioPlayer.duration
                : (currentRecording?.duration_seconds || 86400);
            const target = Math.max(0, Math.min(sec, maxDur));
            audioPlayer.currentTime = target;
            const playPromise = audioPlayer.play();
            if (playPromise !== undefined) {
                playPromise.catch(err => {
                    console.warn("Playback error or autoplay policy:", err);
                });
            }
            onMediaTimeUpdate();
        };

        if (audioPlayer.readyState >= 1) {
            applySeekAndPlay();
        } else {
            pendingSeekHandler = () => {
                pendingSeekHandler = null;
                applySeekAndPlay();
            };
            audioPlayer.addEventListener("loadedmetadata", pendingSeekHandler, { once: true });
            if (currentRecording?.id && !audioPlayer.getAttribute("src")) {
                audioPlayer.src = `/api/recordings/${currentRecording.id}/audio`;
                audioPlayer.load();
            }
        }
    };

    window.seekEvidence = function(startSec, segmentId) {
        seekMedia(startSec);
        let targetTurn = null;
        if (segmentId !== null && segmentId !== undefined) {
            targetTurn = document.querySelector(`.transcript-turn[data-segment-id="${segmentId}"]`) ||
                         document.querySelector(`.transcript-turn[data-segment-index="${segmentId}"]`);
        }
        if (!targetTurn) {
            const turns = document.querySelectorAll(".transcript-turn");
            for (const t of turns) {
                const s = parseFloat(t.getAttribute("data-start") || 0);
                const e = parseFloat(t.getAttribute("data-end") || 0);
                if (startSec >= (s - 0.75) && startSec <= (e + 0.75)) {
                    targetTurn = t;
                    break;
                }
            }
        }
        if (targetTurn) {
            document.querySelectorAll(".transcript-turn.active-speaking").forEach(el => el.classList.remove("active-speaking"));
            targetTurn.classList.add("active-speaking");
            targetTurn.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    };

    // =========================================================================
    // Staged File Management & Manual Start Handler
    // =========================================================================
    function stageFile(file) {
        if (!file) return;
        stagedFile = file;
        if (stagedFileName) stagedFileName.textContent = file.name;
        const isVideo = /\.(mp4|webm|mov)$/i.test(file.name);
        if (stagedFileIcon) {
            stagedFileIcon.innerHTML = isVideo
                ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`
                : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;
        }
        const sizeStr = formatBytes(file.size);
        if (stagedFileMeta) stagedFileMeta.textContent = `${sizeStr} • Attached & ready • Review context hint above, then click start or press Enter`;
        if (stagedFileCard) stagedFileCard.classList.remove("hidden");
        updateSubmitBtnLabel();
        if (submitProcessBtn && document.activeElement !== uploadTopic) {
            submitProcessBtn.focus({ preventScroll: true });
        }
        showAlert(`File attached: <strong>${file.name}</strong>. You can now adjust the topic/context hint and click <strong>Start Processing</strong> (or press Enter).`, "info");
    }

    function clearStagedFile() {
        stagedFile = null;
        if (fileInput) fileInput.value = "";
        if (stagedFileCard) stagedFileCard.classList.add("hidden");
    }

    function updateSubmitBtnLabel() {
        if (!submitProcessBtn) return;
        const typeVal = uploadRecordingType ? uploadRecordingType.value : "Lecture";
        submitProcessBtn.innerHTML = typeVal === "Meeting"
            ? `<span>Start Processing Meeting</span>`
            : `<span>Start Processing Lecture</span>`;
    }

    function triggerStagedProcessing() {
        if (!stagedFile) {
            showAlert("Please select or drop an audio or video file first.", "warning");
            if (fileInput) fileInput.click();
            return;
        }
        if (isPipelineBusy) {
            showAlert("Processing is already in progress. Please wait for the current stage to complete.", "info");
            alertBox?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
            return;
        }
        const fileToUpload = stagedFile;
        stagedFile = null;
        if (stagedFileCard) stagedFileCard.classList.add("hidden");
        handleFileUpload(fileToUpload);
    }

    // =========================================================================
    // File Upload Handler
    // =========================================================================
    function handleFileUpload(file) {
        if (isPipelineBusy) {
            showAlert("Processing is already in progress. Please wait for the current stage to complete.", "info");
            alertBox?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
            return;
        }
        hideAlert();
        switchView("uploads");

        const formData = new FormData();
        formData.append("file", file);

        const topicInput = document.getElementById("uploadTopic");
        const typeSelect = document.getElementById("uploadRecordingType");
        const topicVal = topicInput ? topicInput.value.trim() : "";
        const typeVal = typeSelect ? typeSelect.value : "Lecture";

        if (topicVal) {
            formData.append("topic", topicVal);
        }
        formData.append("recording_type", typeVal);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/upload", true);

        isPipelineBusy = true;
        resetStepperRun();
        updateStepper("upload", 15, `Uploading & validating ${file.name}...`);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.min(35, Math.round((e.loaded / e.total) * 35));
                updateStepper("upload", percent, `Uploading ${file.name}...`);
            }
        };

        xhr.onload = async () => {
            if (fileInput) fileInput.value = "";
            let response;
            try {
                response = JSON.parse(xhr.responseText);
            } catch (err) {
                progressContainer.classList.add("hidden");
                isPipelineBusy = false;
                showAlert("Unexpected server response format.", "error");
                return;
            }

            if (xhr.status === 201 && response.status === "success") {
                const rec = response.data;
                showAlert(`Uploaded successfully: ${rec.filename}. Starting intelligence pipeline...`, "info");
                await loadRecordings(false);
                await startProcessing(rec.id, false, true);
            } else {
                progressContainer.classList.add("hidden");
                const errorMsg = response.message || (response.detail ? JSON.stringify(response.detail) : "Upload validation failed.");
                isPipelineBusy = false;
                showAlert(`<strong>${response.code || "VALIDATION_ERROR"}:</strong> ${errorMsg}`, "error");
            }
        };

        xhr.onerror = () => {
            if (fileInput) fileInput.value = "";
            progressContainer.classList.add("hidden");
            isPipelineBusy = false;
            showAlert("Network error occurred while connecting to the server.", "error");
            alertBox?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
        };

        xhr.onabort = () => {
            if (fileInput) fileInput.value = "";
            progressContainer.classList.add("hidden");
            isPipelineBusy = false;
            showAlert("Upload was cancelled.", "warning");
        };

        xhr.send(formData);
    }

    function updateStepper(activeStepKey, percent, label) {
        if (!progressContainer) return;
        const _stepOrder = ["upload", "preprocess", "transcribe", "normalize", "intelligence", "ready"];
        let _reqIdx = _stepOrder.indexOf(activeStepKey);
        if (_reqIdx < stepperRunMaxIdx) {
            _reqIdx = stepperRunMaxIdx;
        } else {
            stepperRunMaxIdx = _reqIdx;
        }
        percent = Math.max(Number(percent) || 0, stepperRunMaxPct);
        stepperRunMaxPct = percent;
        activeStepKey = _stepOrder[Math.max(0, _reqIdx)] || activeStepKey;
        progressContainer.classList.remove("hidden");
        progressBarFill.style.width = `${percent}%`;
        progressPercent.innerText = `${percent}%`;
        progressLabel.innerText = label;

        const stepKeys = ["upload", "preprocess", "transcribe", "normalize", "intelligence", "ready"];
        const activeIdx = stepKeys.indexOf(activeStepKey);

        stepKeys.forEach((key, idx) => {
            const el = stepItems[key];
            if (!el) return;
            el.classList.remove("active", "completed");
            if (idx < activeIdx) {
                el.classList.add("completed");
            } else if (idx === activeIdx) {
                el.classList.add("active");
            }
        });
    }



    // =========================================================================
    // Stage-Aware Processing Pipeline
    // =========================================================================
    window.startProcessing = async function(recordingId, forceFull = false, allowFromUpload = false) {
        if (isPipelineBusy && !allowFromUpload) { showAlert("Processing is already in progress. Please wait for the current stage to complete.", "info"); return; }
        isPipelineBusy = true;
        if (!allowFromUpload) resetStepperRun();
        hideAlert();
        switchView("uploads");
        try {
            let currentStatus = "UPLOADED";
            let hasTranscript = false;
            try {
                const recCheck = await fetch(`/api/recordings/${recordingId}`);
                if (recCheck.ok) {
                    const checkData = await recCheck.json();
                    currentStatus = checkData.status || (checkData.data && checkData.data.status) || "UPLOADED";
                }
                const trCheck = await fetch(`/api/recordings/${recordingId}/transcript`);
                if (trCheck.ok) {
                    const trData = await trCheck.json();
                    if ((trData.segments && trData.segments.length > 0) || (trData.data && trData.data.segments && trData.data.segments.length > 0)) {
                        hasTranscript = true;
                    }
                }
            } catch (e) {
                console.warn("Could not check status before processing:", e);
            }

            if (currentStatus === "COMPLETED" && !forceFull) {
                isPipelineBusy = false; clearActiveStageTimer(); showAlert("Recording is already processed. Opening dashboard...", "success");
                if (progressContainer) progressContainer.classList.add("hidden");
                await openDashboard(recordingId);
                return;
            }

            // Stage 1: Audio Extraction & Resampling
            if (forceFull || currentStatus === "UPLOADED" || currentStatus === "FAILED") {
                updateStepper("preprocess", 20, "Stage 1/5: Extracting audio & resampling to 16kHz mono...");
                const preRes = await fetch(`/api/recordings/${recordingId}/preprocess`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" }
                });
                if (!preRes.ok) {
                    const err = await preRes.json().catch(() => ({}));
                    throw new Error(err.detail || `Preprocessing failed (HTTP ${preRes.status})`);
                }
            } else {
                updateStepper("preprocess", 20, "Stage 1/5: Audio already preprocessed (skipping)...");
            }

            // Stage 2: Silence-Aware Intelligent Chunking
            if (forceFull || (!hasTranscript && (currentStatus === "UPLOADED" || currentStatus === "PREPROCESSED" || currentStatus === "FAILED"))) {
                updateStepper("preprocess", 35, "Stage 2/5: Partitioning into silence-aware chunks...");
                const chunkRes = await fetch(`/api/recordings/${recordingId}/chunk`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ target_duration: 25.0, max_duration: 30.0, overlap_seconds: 0.5 })
                });
                if (!chunkRes.ok) {
                    const err = await chunkRes.json().catch(() => ({}));
                    throw new Error(err.detail || `Audio chunking failed (HTTP ${chunkRes.status})`);
                }
            } else {
                updateStepper("preprocess", 35, "Stage 2/5: Chunks already partitioned (skipping)...");
            }

            // Stage 3: Whisper ASR & Roman Urdu Normalization
            if (forceFull || (!hasTranscript && currentStatus !== "TRANSCRIBED" && currentStatus !== "COMPLETED")) {
                updateStepper("transcribe", 55, "Stage 3/5: Whisper Speech-to-Text & Roman Urdu normalization in progress...");

                // Dynamic ticker so the user sees live execution during CPU decoding
                let transcribeElapsedSec = 0;
                const transcribeTimer = startActiveStageTimer(() => {
                    transcribeElapsedSec += 2;
                    const mins = Math.floor(transcribeElapsedSec / 60);
                    const secs = transcribeElapsedSec % 60;
                    const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                    const p = Math.min(75, 55 + Math.floor(transcribeElapsedSec / 15));
                    updateStepper("transcribe", p, `Stage 3/5: Whisper Speech-to-Text & Roman Urdu normalization (${timeStr} elapsed, running on CUDA GPU)...`);
                }, 2000);

                let trRes;
                try {
                    trRes = await fetch(`/api/recordings/${recordingId}/transcribe`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ model_size: "base" })
                    });
                } finally {
                    clearActiveStageTimer();
                }

                if (!trRes.ok) {
                    const err = await trRes.json().catch(() => ({}));
                    throw new Error(err.detail || `Transcription failed (HTTP ${trRes.status})`);
                }
            } else {
                updateStepper("transcribe", 65, "Stage 3/5: Transcription already completed (skipping)...");
            }

            // Stage 4: Structured Lecture Intelligence Extraction
            updateStepper("normalize", 72, "Stage 3b/5: Roman Urdu text normalization & translation verified...");
            updateStepper("intelligence", 80, "Stage 4/5: Extracting summaries, notes, assignments & quizzes...");
            let intelElapsedSec = 0;
            const intelTimer = startActiveStageTimer(() => {
                intelElapsedSec += 2;
                const mins = Math.floor(intelElapsedSec / 60);
                const secs = intelElapsedSec % 60;
                const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                const p = Math.min(94, 80 + Math.floor(intelElapsedSec / 20));
                let phase = "Extracting topics, definitions & formulas";
                if (intelElapsedSec > 90 && intelElapsedSec <= 180) {
                    phase = "Synthesizing executive briefing & structured study notes";
                } else if (intelElapsedSec > 180) {
                    phase = "Generating practice quizzes & exam recall questions";
                }
                updateStepper("intelligence", p, `Stage 4/5: ${phase} (${timeStr} elapsed, running AI intelligence on CUDA GPU)...`);
            }, 2000);

            let intelRes;
            try {
                intelRes = await fetch(`/api/recordings/${recordingId}/intelligence/generate`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" }
                });
            } finally {
                clearActiveStageTimer();
            }

            if (!intelRes.ok) {
                const err = await intelRes.json().catch(() => ({}));
                throw new Error(err.detail || `Intelligence generation failed (HTTP ${intelRes.status})`);
            }

            // Stage 5: Dense Multilingual Vector Embeddings
            updateStepper("ready", 95, "Stage 5/5: Generating dense vector embeddings for semantic search & Q&A...");
            let embElapsedSec = 0;
            const embTimer = startActiveStageTimer(() => {
                embElapsedSec += 2;
                const p = Math.min(99, 95 + Math.floor(embElapsedSec / 5));
                updateStepper("ready", p, `Stage 5/5: Generating dense vector embeddings (${embElapsedSec}s elapsed)...`);
            }, 2000);

            try {
                const embRes = await fetch(`/api/recordings/${recordingId}/embeddings/generate`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" }
                });
                if (!embRes.ok) {
                    console.warn("Embedding generation returned non-200, continuing with lexical QA fallback.");
                }
            } catch (embErr) {
                console.warn("Embedding generation network error, continuing:", embErr);
            } finally {
                clearActiveStageTimer();
            }

            // Complete
            isPipelineBusy = false; clearActiveStageTimer(); updateStepper("ready", 100, "Intelligence processing complete!");
            showAlert(`Pipeline complete! Loading lecture workspace...`, "success");

            await loadRecordings(false);

            setTimeout(() => {
                if (progressContainer) progressContainer.classList.add("hidden");
                openDashboard(recordingId);
            }, 600);

        } catch (err) {
            if (progressContainer) progressContainer.classList.add("hidden");
            isPipelineBusy = false; clearActiveStageTimer(); showAlert(`<strong>Processing Failed:</strong> ${err.message}`, "error");
            await loadRecordings(false);
        }
    };

    // =========================================================================
    // Open & Render Dashboard
    // =========================================================================
    async function openDashboard(recordingId) {
        hideAlert();
        if (progressContainer) progressContainer.classList.add("hidden");
        try {
            const isDifferentAudio = (lastDashboardRecordingId !== recordingId);
            lastDashboardRecordingId = recordingId;

            // 1. Fetch recording metadata
            const recRes = await fetch(`/api/recordings/${recordingId}`);
            if (!recRes.ok) {
                showAlert(`Failed to load recording metadata (HTTP ${recRes.status}).`, "error");
                return;
            }
            const recData = await recRes.json();
            currentRecording = recData.data || recData;

            // Switch view immediately to dashboard so user sees instant responsiveness
            switchView("dashboard");
            switchWorkspaceTab(activeWorkspaceTab || "overview");
            window.scrollTo({ top: 0, behavior: "smooth" });

            // 2. Fetch intelligence report
            let intelData = {};
            try {
                const getRes = await fetch(`/api/recordings/${recordingId}/intelligence`);
                if (getRes.ok) {
                    intelData = await getRes.json();
                }
            } catch (e) {
                console.warn("Could not fetch intelligence:", e);
            }
            currentIntelligence = (intelData && intelData.data) ? intelData.data : intelData;

            // 3. Fetch transcripts
            let trSegments = [];
            try {
                const trRes = await fetch(`/api/recordings/${recordingId}/transcript`);
                if (trRes.ok) {
                    const trData = await trRes.json();
                    trSegments = trData.segments || (trData.data && trData.data.segments) || [];
                }
            } catch (e) {
                console.warn("Could not fetch transcript:", e);
            }
            currentSegments = trSegments;

            // Render workspace defensively
            try {
                renderDashboard(isDifferentAudio);
            } catch (renderErr) {
                console.error("Dashboard render warning:", renderErr);
            }

        } catch (err) {
            console.error("openDashboard error:", err);
            showAlert(`Failed to open dashboard: ${err.message}`, "error");
        }
    }
    window.openDashboard = openDashboard;

    function renderDashboard(isDifferentAudio = true) {
        const rec = currentRecording;
        if (!rec) return;
        const intel = currentIntelligence || {};

        // 1. Hero Details
        const titleEl = document.getElementById("dashboardRecordingTitle");
        if (titleEl) titleEl.innerText = (rec.filename || "Lecture Recording").replace(/\.[^/.]+$/, "");

        const durEl = document.getElementById("dashboardDuration");
        if (durEl) durEl.innerText = formatDurationExtended(rec.duration_seconds || 0);

        const fmtEl = document.getElementById("dashboardFormatSize");
        if (fmtEl) {
            const fmtName = (rec.file_format || (rec.filename || '').split('.').pop() || 'MEDIA').toUpperCase().replace(".", "");
            fmtEl.innerText = `${fmtName} (${formatBytes(rec.file_size_bytes || 0)})`;
        }

        const dateEl = document.getElementById("dashboardDate");
        if (dateEl) dateEl.innerText = new Date(rec.created_at || Date.now()).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });

        // Domain Chip
        const domainChip = document.getElementById("dashboardDomainChip");
        if (domainChip) {
            const detectedDomain = (intel.lecture_notes?.topics && intel.lecture_notes.topics[0]) || ((rec.filename || '').toLowerCase().includes("operating") ? "Operating Systems" : "Computer Science");
            domainChip.innerText = detectedDomain;
        }

        // Status Badge
        const statusBadge = document.getElementById("dashboardStatusBadge");
        if (statusBadge) {
            const statusUpper = (rec.status || 'UNKNOWN').toUpperCase();
            statusBadge.className = `status-badge-chip ${statusUpper === 'COMPLETED' ? 'badge-success' : (statusUpper === 'FAILED' ? 'badge-danger' : 'badge-warning')}`;
            statusBadge.innerText = statusUpper;
        }

        // Process / Reprocess Button (Always Visible in Lecture Hero Bar)
        const processBtn = document.getElementById("dashboardProcessBtn");
        if (processBtn) {
            processBtn.style.display = "inline-flex";
            if ((rec.status || '').toUpperCase() === "COMPLETED") {
                processBtn.className = "btn btn-secondary btn-sm";
                processBtn.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <polyline points="23 4 23 10 17 10"></polyline>
                        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                    </svg>
                    <span>Re-process Recording</span>
                `;
                processBtn.title = "Re-process or re-analyze this recording";
                processBtn.onclick = () => openReprocessModal(rec.id, rec.filename);
            } else {
                processBtn.className = "btn btn-success btn-sm";
                processBtn.innerHTML = `<span>Process Recording</span>`;
                processBtn.title = "Start processing recording";
                processBtn.onclick = () => startProcessing(rec.id, false);
            }
        }

        // Media Stream Setup (always use simple compact audio player)
        const audioSrc = `/api/recordings/${rec.id}/audio`;
        if (audioPlayer) {
            audioPlayer.style.display = "block";
            if (isDifferentAudio || audioPlayer.dataset.recordingId !== String(rec.id)) {
                try {
                    audioPlayer.src = audioSrc;
                    audioPlayer.dataset.recordingId = String(rec.id);
                    audioPlayer.load();
                } catch (e) {
                    console.warn("Could not load audio player source:", e);
                }
            }
        }
        if (audioVisualizerBlock) audioVisualizerBlock.style.display = "flex";

        // Chapter Pins on Waveform
        renderChapterPins(rec.duration_seconds, intel.lecture_notes);

        // 2. Executive Summary
        document.getElementById("summaryShort").innerText = intel.summary_short || "No executive summary available for this recording.";
        
        // Topic Keyword Chips with Grouped Hierarchy (Topics vs Core Concepts) & Interactive Seeking
        const summaryChipsEl = document.getElementById("summaryTopicChips");
        if (summaryChipsEl) {
            const studyTopics = intel.lecture_notes?.study_topics || [];
            const topicChips = [];
            const conceptChips = [];
            const seenNorms = new Set();
            const fillerRegex = /^(?:so\s+)?here\s+are\s+(?:some\s+)?(?:important\s+)?distinctions?|^(?:so|and|then|also|now|here|as|let|we|you)\b|\b(?:distinctions?|overview)\.?$/i;

            // 1. Structured Lecture Topics (with direct link to topic note card)
            studyTopics.forEach((t, idx) => {
                const title = t.title ? t.title.trim() : "";
                const norm = title.toLowerCase().replace(/[^a-z0-9]/g, "");
                if (norm && !seenNorms.has(norm)) {
                    seenNorms.add(norm);
                    const startSec = t.start_time !== undefined ? t.start_time : (t.spans && t.spans[0] ? t.spans[0].start : 0);
                    topicChips.push({
                        idx: idx + 1,
                        label: title,
                        time: startSec
                    });
                }
            });

            // 2. Core Concepts / Terms Discussed Inside Topics
            const concepts = intel.lecture_notes?.key_concepts || [];
            concepts.forEach(c => {
                const text = typeof c === "string" ? c.trim().replace(/[.:,;]+$/, "") : "";
                if (!text || text.length < 3 || text.length > 45 || fillerRegex.test(text)) return;
                const norm = text.toLowerCase().replace(/[^a-z0-9]/g, "");
                if (norm && !seenNorms.has(norm)) {
                    seenNorms.add(norm);
                    const matchingTopic = studyTopics.find(t =>
                        (t.key_points && t.key_points.some(kp => kp.toLowerCase().includes(text.toLowerCase()))) ||
                        (t.definitions && t.definitions.some(d => d.term && d.term.toLowerCase() === text.toLowerCase())) ||
                        (t.explanation && t.explanation.toLowerCase().includes(text.toLowerCase()))
                    );
                    const seekTime = matchingTopic ? (matchingTopic.start_time || 0) : 0;
                    conceptChips.push({
                        label: text,
                        time: seekTime
                    });
                }
            });

            let html = "";
            if (topicChips.length > 0) {
                html += `
                    <div style="width:100%; margin-bottom:0.75rem;">
                        <div style="font-size:0.72rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.4rem; display:flex; align-items:center; gap:5px;">
                            <span>Lecture Chapters (${topicChips.length})</span>
                        </div>
                        <div style="display:flex; flex-wrap:wrap; gap:0.4rem;">
                            ${topicChips.map(tc => `
                                <button type="button" class="topic-tag-chip" onclick="seekMedia(${tc.time}); switchWorkspaceTab('notes');" title="Jump to Topic ${tc.idx} in Study Notes (${formatDurationSimple(tc.time)})" style="cursor:pointer; display:inline-flex; align-items:center; gap:5px; background:var(--primary-light); color:var(--primary); border-color:var(--primary-border); font-weight:700;">
                                    <span>${tc.idx}. ${tc.label}</span>
                                    <span style="font-size:0.72rem; font-family:var(--font-mono); opacity:0.85;">${formatDurationSimple(tc.time)}</span>
                                </button>
                            `).join("")}
                        </div>
                    </div>
                `;
            }

            if (conceptChips.length > 0) {
                html += `
                    <div style="width:100%;">
                        <div style="font-size:0.72rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.4rem; display:flex; align-items:center; gap:5px;">
                            <span>Core Concepts & Formulas</span>
                        </div>
                        <div style="display:flex; flex-wrap:wrap; gap:0.4rem;">
                            ${conceptChips.slice(0, 4).map(cc => `
                                <button type="button" class="topic-tag-chip" onclick="seekMedia(${cc.time}); switchWorkspaceTab('notes');" title="Seek to ${formatDurationSimple(cc.time)} and view relevant section" style="cursor:pointer; display:inline-flex; align-items:center; gap:5px;">
                                    <span>${cc.label}</span>
                                    <span style="font-size:0.72rem; color:var(--primary); font-family:var(--font-mono); font-weight:700;">${formatDurationSimple(cc.time)}</span>
                                </button>
                            `).join("")}
                        </div>
                    </div>
                `;
            }

            summaryChipsEl.innerHTML = html || `<span class="topic-tag-chip">General Academic Lecture</span>`;
        }

        // 3. Topic-wise Study Notes
        renderTopicNotes(intel.lecture_notes || {});

        // 4. Assignments & Action Items
        renderAssignments(intel.assignments || []);
        renderActionItems(intel.action_items || []);

        // 5. Practice Quizzes Carousel
        quizQuestions = intel.quizzes || [];
        currentQuizIdx = 0;
        renderActiveQuizQuestion();

        // 6. Transcripts Feed & Timeline
        renderTranscripts();
        renderTimelineFeed();
        renderHighlightsFeed();

        // 7. Refresh Chatbot Chat for opened audio dashboard
        if (isDifferentAudio) {
            refreshQAChat(rec, intel);
        }

        // 8. Overview Deliverables Preview & Metrics
        const previewEl = document.getElementById("overviewAssignmentsPreview");
        const asgns = intel.assignments || [];
        const actions = intel.action_items || [];
        if (previewEl) {
            if (asgns.length > 0) {
                previewEl.innerHTML = asgns.slice(0, 2).map(a => `
                    <div class="overview-preview-item">
                        <div>
                            <div class="preview-title"><span class="mono-emoji indigo">📋</span>${a.topic || 'Coursework Assignment'}</div>
                            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;">${a.required_tasks || a.description || ''}</div>
                        </div>
                        <span class="preview-due">Due: ${a.deadline || 'Specified in audio'}</span>
                    </div>
                `).join("");
            } else if (actions.length > 0) {
                previewEl.innerHTML = actions.slice(0, 2).map(ai => `
                    <div class="overview-preview-item">
                        <div>
                            <div class="preview-title"><span class="mono-emoji emerald">✅</span>Task: ${ai.task}</div>
                        </div>
                        <span class="preview-due">Due: ${ai.deadline || 'Pending'}</span>
                    </div>
                `).join("");
            } else {
                previewEl.innerHTML = `<div class="empty-state" style="padding:0.75rem;">No upcoming assignments or action items.</div>`;
            }
        }
    }

    // =========================================================================
    // Topic Notes Renderer — v2 (real semantic topics with grounded timestamps)
    // =========================================================================
    function renderTopicNotes(notes) {
        const container = document.getElementById("topicNotesContainer");
        const studyTopics = notes.study_topics || [];
        const analyzerUsed = notes.analyzer_used || "local";
        const legacyTopics = notes.topics || [];

        if (studyTopics.length === 0 && legacyTopics.length === 0) {
            container.innerHTML = `<div class="empty-state">No structured topic notes extracted. Regenerate the intelligence report to analyse this lecture.</div>`;
            return;
        }

        let badgeColor = "#10B981";
        let badgeLabel = "Local AI (Qwen3.5 4B)";
        let badgeTitle = "Generated using Qwen3.5 4B via Ollama";

        if (notes.is_fallback || analyzerUsed === "local_extractive_fallback") {
            badgeColor = "#D97706"; // Amber-600
            badgeLabel = "Extractive Fallback";
            badgeTitle = notes.fallback_reason ? `Extractive Fallback: ${notes.fallback_reason}` : "Generated using deterministic extractive NLP fallback engine";
        } else if (analyzerUsed === "gemini" || analyzerUsed === "cloud_llm") {
            badgeColor = "#6366F1";
            badgeLabel = "Cloud AI";
            badgeTitle = "Generated using cloud LLM";
        }

        let html = `
            <div style="display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:0.75rem; flex-wrap:wrap;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="background:${badgeColor}; color:#fff; font-size:0.72rem; padding:2px 8px; border-radius:12px; font-weight:600; display:inline-block;" title="${badgeTitle}">${badgeLabel}</span>
                    ${(notes.is_fallback || analyzerUsed === "local_extractive_fallback") && notes.fallback_reason ? `<span style="font-size:0.72rem; color:#D97706; background:#FEF3C7; padding:2px 8px; border-radius:6px; border:1px solid #FDE68A;">${notes.fallback_reason}</span>` : ""}
                </div>
                ${notes.markdown_study_guide ? `<button onclick="window.openMasterStudyGuideModal()" style="background:#EEF2FF; color:#4F46E5; border:1px solid #C7D2FE; border-radius:6px; font-size:0.75rem; font-weight:700; padding:3px 9px; cursor:pointer;">View Full Study Guide →</button>` : ""}
            </div>
        `;

        if (studyTopics.length > 0) {
            const conceptRels = notes.concept_relationships || [];
            if (notes.lecture_overview || conceptRels.length > 0) {
                const relHtml = conceptRels.length > 0
                    ? `<div style="margin-top:0.45rem; font-size:0.77rem; color:#3730A3;"><strong>How Concepts Connect:</strong> ${conceptRels.slice(0, 4).map(r => `<span>${r.from_topic} &rarr; <em>${r.relation}</em> &rarr; ${r.to_topic}</span>`).join(" &nbsp;|&nbsp; ")}</div>`
                    : "";
                html += `
                    <div style="background:#F8FAFF; border:1px solid #E0E7FF; border-left:3px solid #4F46E5; border-radius:8px; padding:0.65rem 0.9rem; margin-bottom:0.85rem; font-size:0.82rem; color:#1E293B; line-height:1.45;">
                        ${notes.lecture_overview ? `<div><strong>Big Picture:</strong> ${notes.lecture_overview}</div>` : ""}
                        ${relHtml}
                    </div>
                `;
            }
            studyTopics.forEach((topic, idx) => {
                const spans = topic.spans || [];
                const startSec = spans.length > 0 ? spans[0].start : (topic.start_time || 0);
                const endSec = spans.length > 0 ? spans[spans.length - 1].end : (topic.end_time || 0);

                let spanChips = "";
                if (spans.length > 1) {
                    spanChips = spans.map(sp =>
                        `<button class="topic-timestamp-range" onclick="seekMedia(${sp.start})">${formatDurationSimple(sp.start)}–${formatDurationSimple(sp.end)}</button>`
                    ).join(" ");
                } else {
                    spanChips = `<button class="topic-timestamp-range" onclick="seekMedia(${startSec})">${formatDurationSimple(startSec)}–${formatDurationSimple(endSec)}</button>`;
                }

                const cleanProse = (val) => {
                    if (!val) return "";
                    if (Array.isArray(val)) return val.map(x => String(x).replace(/^[\s,'"\n]+|[\s,'"\n]+$/g, "")).filter(Boolean).join(" ");
                    let s = String(val).trim();
                    if (s.startsWith("[") && s.endsWith("]")) {
                        const parts = [];
                        const re = /['"]([^'"]+)['"]/g;
                        let m;
                        while ((m = re.exec(s)) !== null) {
                            const seg = m[1].replace(/\\n/g, " ").replace(/^[\s,]+|[\s,]+$/g, "").trim();
                            if (seg && /[a-zA-Z0-9]{2,}/.test(seg)) parts.push(seg);
                        }
                        if (parts.length > 0) return parts.join(" ");
                    }
                    return s;
                };
                const isDupNote = (cand, seenList) => {
                    const nc = (cand || "").toLowerCase().replace(/[^a-z0-9]/g, "");
                    if (!nc) return true;
                    const tc = new Set((cand || "").toLowerCase().split(/\W+/).filter(w => w.length > 2));
                    for (const prev of seenList) {
                        const np = (prev || "").toLowerCase().replace(/[^a-z0-9]/g, "");
                        if (!np) continue;
                        if (nc === np || (nc.length >= 22 && (nc.includes(np) || np.includes(nc)))) return true;
                        const tp = new Set((prev || "").toLowerCase().split(/\W+/).filter(w => w.length > 2));
                        if (tc.size > 0 && tp.size > 0) {
                            let inter = 0;
                            tc.forEach(w => { if (tp.has(w)) inter++; });
                            if (inter / Math.min(tc.size, tp.size) >= 0.72) return true;
                        }
                    }
                    return false;
                };

                const shownCardTexts = [];
                const cleanExpl = cleanProse(topic.explanation) || "Key concepts discussed in this section.";
                shownCardTexts.push(cleanExpl);

                const rawWhat = cleanProse(topic.what_it_means);
                const whatHtml = (rawWhat && !isDupNote(rawWhat, shownCardTexts))
                    ? (shownCardTexts.push(rawWhat), `<div style="margin-top:0.35rem; font-size:0.8rem; color:#334155;"><strong>In Plain Words:</strong> ${rawWhat}</div>`)
                    : "";

                const rawHow = cleanProse(topic.how_it_works);
                const howHtml = (rawHow && !isDupNote(rawHow, shownCardTexts))
                    ? (shownCardTexts.push(rawHow), `<div style="margin-top:0.3rem; font-size:0.8rem; color:#334155;"><strong>How It Works:</strong> ${rawHow}</div>`)
                    : "";

                const rawSteps = Array.isArray(topic.steps_and_logic) ? topic.steps_and_logic : [];
                const cleanSteps = [];
                for (const st of rawSteps) {
                    const cs = cleanProse(st);
                    if (cs && !isDupNote(cs, shownCardTexts)) {
                        shownCardTexts.push(cs);
                        cleanSteps.push(cs);
                    }
                    if (cleanSteps.length >= 4) break;
                }
                const stepsHtml = cleanSteps.length > 0
                    ? `<div style="margin-top:0.35rem; font-size:0.79rem; color:#334155;"><strong>Step-by-Step Breakdown:</strong><ol style="margin:0.2rem 0 0 1.2rem; padding:0;">${cleanSteps.map(st => `<li style="margin-bottom:2px;">${st}</li>`).join("")}</ol></div>`
                    : "";

                const rawWhy = cleanProse(topic.why_it_matters);
                const whyHtml = (rawWhy && !isDupNote(rawWhy, shownCardTexts))
                    ? (shownCardTexts.push(rawWhy), `<div style="margin-top:0.3rem; font-size:0.79rem; color:#0F766E;"><strong>Why It Matters:</strong> ${rawWhy}</div>`)
                    : "";

                const keyPoints = [];
                for (const kp of (topic.key_points || [])) {
                    const ckp = cleanProse(kp);
                    if (ckp && !isDupNote(ckp, shownCardTexts)) {
                        shownCardTexts.push(ckp);
                        keyPoints.push(ckp);
                    }
                    if (keyPoints.length >= 5) break;
                }
                const keyPointsHtml = keyPoints.length > 0
                    ? `<ul class="topic-key-points">${keyPoints.map(kp => `<li>${kp}</li>`).join("")}</ul>`
                    : "";

                const examples = topic.examples || [];
                const examplesHtml = examples.length > 0
                    ? `<div class="topic-examples"><strong>Example:</strong> ${examples[0]}</div>`
                    : "";

                const formulas = topic.formulas || [];
                const formulasHtml = formulas.length > 0
                    ? formulas.map(f => `<span class="topic-formula-pill">${f}</span>`).join(" ")
                    : "";

                const defs = topic.definitions || [];
                const defsHtml = defs.length > 0
                    ? `<div style="margin-top:0.35rem; font-size:0.78rem; color:#475569;">${defs.slice(0, 3).map(d => `<span style="background:#F1F5F9; border:1px solid #E2E8F0; padding:2px 7px; border-radius:6px; margin-right:4px; display:inline-block; margin-bottom:3px;"><strong>${d.term}:</strong> ${d.definition}</span>`).join(" ")}</div>`
                    : "";

                const emphasis = topic.emphasis || [];
                const emphasisHtml = emphasis.length > 0
                    ? `<div style="margin-top:0.35rem; font-size:0.75rem;">${emphasis.slice(0, 2).map(e => `<span style="background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; padding:2px 7px; border-radius:6px; margin-right:4px; font-weight:600; display:inline-block; margin-bottom:3px;">${e.text}</span>`).join(" ")}</div>`
                    : "";

                const examBadge = topic.exam_relevant
                    ? `<span class="exam-relevant-badge"><span class="mono-emoji amber">🎯</span>Exam Relevant</span>`
                    : "";

                const evidenceList = topic.evidence || (topic.source_segment_ids ? topic.source_segment_ids.map(sid => `seg_${String(sid).padStart(3, '0')}`) : []);
                let evidenceBadge = "";
                if (evidenceList.length > 0) {
                    const evText = evidenceList.length === 1 ? evidenceList[0] : `${evidenceList[0]}–${evidenceList[evidenceList.length - 1]}`;
                    evidenceBadge = `<span class="evidence-badge" title="Grounded in transcript segments"><span class="mono-emoji green">✓</span> Grounded in ${evText}</span>`;
                }

                const subtopics = topic.subtopics || [];
                let subtopicsHtml = "";
                if (subtopics.length > 0) {
                    subtopicsHtml = `
                        <div class="topic-subtopics-section">
                            <div class="topic-subtopics-header">Subtopics & Concept Hierarchy</div>
                            <div class="topic-subtopics-list">
                                ${subtopics.map(st => {
                                    let roleBadgeClass = "role-subtopic";
                                    let roleName = "Subtopic";
                                    if (st.role === "DEFINITION") { roleBadgeClass = "role-def"; roleName = "Definition"; }
                                    else if (st.role === "EXAMPLE") { roleBadgeClass = "role-ex"; roleName = "Example"; }
                                    else if (st.role === "ASSIGNMENT") { roleBadgeClass = "role-assign"; roleName = "Directive"; }
                                    const evTag = st.source_segment_ids && st.source_segment_ids.length > 0
                                        ? `<span class="subtopic-evidence">seg_${String(st.source_segment_ids[0]).padStart(3, '0')}</span>`
                                        : (st.evidence && st.evidence.length > 0 ? `<span class="subtopic-evidence">${st.evidence[0]}</span>` : '');
                                    return `
                                        <div class="subtopic-item">
                                            <span class="subtopic-role-badge ${roleBadgeClass}">${roleName}</span>
                                            <span class="subtopic-title">${st.title}</span>
                                            ${evTag}
                                        </div>
                                    `;
                                }).join("")}
                            </div>
                        </div>
                    `;
                }

                const aiBackgroundHtml = topic.ai_background_enhancement
                    ? `<div class="ai-background-callout"><strong>AI Conceptual Enrichment:</strong> ${topic.ai_background_enhancement}</div>`
                    : "";

                html += `
                    <div class="topic-note-item">
                        <div class="topic-number-badge">${idx + 1}</div>
                        <div class="topic-note-content">
                            <div class="topic-note-header">
                                <span class="topic-note-title">${topic.title}</span>
                                <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
                                    ${evidenceBadge}
                                    ${examBadge}
                                </div>
                            </div>
                            <div class="topic-span-chips">${spanChips}</div>
                            <p class="topic-note-desc">${cleanExpl}</p>
                            ${whatHtml}
                            ${howHtml}
                            ${stepsHtml}
                            ${whyHtml}
                            ${keyPointsHtml}
                            ${defsHtml}
                            ${examplesHtml}
                            ${formulasHtml ? `<div class="topic-formulas">${formulasHtml}</div>` : ""}
                            ${emphasisHtml}
                            ${subtopicsHtml}
                            ${aiBackgroundHtml}
                        </div>
                    </div>
                `;
            });
        } else {
            legacyTopics.forEach((topic, idx) => {
                html += `
                    <div class="topic-note-item">
                        <div class="topic-number-badge">${idx + 1}</div>
                        <div class="topic-note-content">
                            <div class="topic-note-header"><span class="topic-note-title">${topic}</span></div>
                            <p class="topic-note-desc">Core concepts discussed in this section.</p>
                        </div>
                    </div>
                `;
            });
        }

        container.innerHTML = html;
    }

    // =========================================================================
    // Timeline Feed Renderer — v2 (real semantic topic timestamps)
    // =========================================================================
    function renderTimelineFeed() {
        const container = document.getElementById("timelineFeedList");
        if (!container) return;
        const notes = currentIntelligence?.lecture_notes || {};
        const studyTopics = notes.study_topics || [];
        const legacyTopics = notes.topics || [];
        const duration = currentRecording?.duration_seconds || 180;

        // Prefer study_topics (v2), fall back to legacy
        if (studyTopics.length === 0 && legacyTopics.length === 0) {
            container.innerHTML = `<div class="empty-state">Timeline milestones will appear after topic discovery.</div>`;
            return;
        }

        let html = "";
        if (studyTopics.length > 0) {
            studyTopics.forEach((t, idx) => {
                const spans = t.spans || [];
                const time = spans.length > 0 ? Math.round(spans[0].start) : Math.round((idx / studyTopics.length) * duration);
                const endTime = spans.length > 0 ? Math.round(spans[spans.length - 1].end) : Math.min(time + 60, duration);
                const kpSnippet = (t.key_points || []).slice(0, 2).join("; ");
                html += `
                    <div class="transcript-turn" data-start="${time}" data-end="${endTime}">
                        <button class="turn-timestamp" onclick="seekMedia(${time})">${formatDurationSimple(time)}</button>
                        <div class="turn-content">
                            <div class="turn-speaker" style="color:var(--primary); font-weight:700;">Topic ${idx + 1}</div>
                            <div class="turn-text"><strong>${t.title}</strong>${kpSnippet ? " — " + kpSnippet : ""}</div>
                        </div>
                    </div>
                `;
            });
        } else {
            legacyTopics.forEach((t, idx) => {
                const time = Math.round((idx / legacyTopics.length) * duration);
                html += `
                    <div class="transcript-turn" data-start="${time}" data-end="${time + 30}">
                        <button class="turn-timestamp" onclick="seekMedia(${time})">${formatDurationSimple(time)}</button>
                        <div class="turn-content">
                            <div class="turn-speaker" style="color:var(--primary); font-weight:700;">Chapter Milestone ${idx + 1}</div>
                            <div class="turn-text"><strong>${t}</strong> — Foundational discussion and key takeaways for this section.</div>
                        </div>
                    </div>
                `;
            });
        }
        container.innerHTML = html;
    }


    // =========================================================================
    // Highlights Feed Renderer (when user clicks Video Highlights tab)
    // =========================================================================
    function renderHighlightsFeed() {
        const container = document.getElementById("highlightsList");
        if (!container) return;
        const pts = currentIntelligence?.lecture_notes?.important_points || [];

        if (pts.length === 0) {
            container.innerHTML = `
                <div class="topic-note-item">
                    <div class="topic-number-badge" style="background:#EEF2FF; color:#4F46E5; font-size:0.75rem; font-weight:700;">Core</div>
                    <div class="topic-note-content">
                        <span class="topic-note-title">Core Discussion</span>
                        <p class="topic-note-desc">Key educational concepts and assignments were emphasized by the speaker.</p>
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = pts.map(p => {
            const seekSec = extractStartSeconds(p.timestamp);
            return `
                <div class="topic-note-item">
                    <div class="topic-number-badge" style="background:${p.is_exam_relevant ? '#FEF3C7' : '#EEF2FF'}; color:${p.is_exam_relevant ? '#B45309' : '#4F46E5'}; font-size:0.75rem; font-weight:700;">
                        ${p.is_exam_relevant ? 'Exam' : 'Key'}
                    </div>
                    <div class="topic-note-content">
                        <div class="topic-note-header">
                            <span class="topic-note-title">${p.is_exam_relevant ? 'Exam-Relevant Concept' : 'Key Highlight'}</span>
                            <button class="turn-timestamp" onclick="seekMedia(${seekSec})">${p.timestamp || 'Recorded'}</button>
                        </div>
                        <p class="topic-note-desc">${p.point}</p>
                    </div>
                </div>
            `;
        }).join("");
    }

    // =========================================================================
    // Assignments & Deadlines Renderer
    // =========================================================================
    function renderAssignments(assignments) {
        const container = document.getElementById("assignmentsContainer");
        if (!assignments || assignments.length === 0) {
            container.innerHTML = `<div class="empty-state" style="padding:1rem;">No assignments detected in this lecture.</div>`;
            return;
        }

        container.innerHTML = assignments.map(a => {
            const seekSec = a.start_time !== undefined ? a.start_time : extractStartSeconds(a.timestamp);
            const isInClass = (a.topic && /in-class|activity|exercise/i.test(a.topic)) || (a.deadline && /in-class|minute/i.test(a.deadline)) || (a.submission_method && /in-class/i.test(a.submission_method));
            const typeBadge = isInClass
                ? `<span style="background:#EEF2FF; color:#4F46E5; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:6px; border:1px solid #C7D2FE;">In-Class Activity</span>`
                : `<span style="background:#F0FDF4; color:#166534; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:6px; border:1px solid #BBF7D0;">Homework Deliverable</span>`;

            let metaChips = [];
            if (a.deadline && a.deadline !== "Not specified") {
                metaChips.push(`<span class="due-chip">Due: ${a.deadline}</span>`);
            }
            if (a.submission_method && a.submission_method !== "Not specified") {
                metaChips.push(`<span class="submission-portal-pill">Submit: ${a.submission_method}</span>`);
            }
            if (a.format && a.format !== "Not specified") {
                metaChips.push(`<span style="background:#F8FAFC; color:#334155; font-size:0.75rem; padding:3px 8px; border-radius:6px; border:1px solid #CBD5E1; font-weight:600;">Format: ${a.format}</span>`);
            }
            if (a.individual_or_group && a.individual_or_group !== "Not specified") {
                metaChips.push(`<span style="background:#F8FAFC; color:#334155; font-size:0.75rem; padding:3px 8px; border-radius:6px; border:1px solid #CBD5E1; font-weight:600;">Team: ${a.individual_or_group}</span>`);
            }
            if (a.required_tools && a.required_tools !== "Not specified") {
                metaChips.push(`<span style="background:#F8FAFC; color:#334155; font-size:0.75rem; padding:3px 8px; border-radius:6px; border:1px solid #CBD5E1; font-weight:600;">Tools: ${a.required_tools}</span>`);
            }

            return `
                <div class="assignment-modern-card" style="border:1px solid #E2E8F0; border-radius:10px; padding:1.1rem; margin-bottom:0.9rem; background:#FFFFFF; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.5rem; flex-wrap:wrap; gap:0.5rem;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            ${typeBadge}
                            <h4 style="font-weight:700; color:#0F172A; font-size:0.96rem; margin:0;">${a.topic || 'Coursework Assignment'}</h4>
                        </div>
                        <button class="turn-timestamp" onclick="seekMedia(${seekSec})" style="background:#EEF2FF; color:#4F46E5; border:none; border-radius:6px; font-size:0.75rem; padding:3px 8px; cursor:pointer; font-weight:700;">${a.timestamp || 'Recorded'}</button>
                    </div>
                    <p class="assignment-card-desc" style="font-size:0.86rem; color:#334155; line-height:1.55; margin-bottom:0.75rem;">${a.required_tasks || a.description || 'Submit required coursework according to the instructions.'}</p>
                    <div class="assignment-meta-row" style="display:flex; flex-wrap:wrap; gap:0.4rem;">
                        ${metaChips.join("")}
                    </div>
                </div>
            `;
        }).join("");
    }

    // =========================================================================
    // Action Items Checklist Renderer
    // =========================================================================
    function renderActionItems(actionItems) {
        const container = document.getElementById("actionItemsContainer");
        if (!actionItems || actionItems.length === 0) {
            container.innerHTML = `<div class="empty-state" style="padding:1rem;">No action items detected.</div>`;
            return;
        }

        container.innerHTML = actionItems.map((ai, idx) => `
            <div class="action-item-row">
                <input type="checkbox" id="action-item-${idx}" class="action-checkbox">
                <label for="action-item-${idx}" class="action-item-label">
                    <strong>${ai.task}</strong>
                    ${ai.deadline ? ` <span style="color:var(--text-muted); font-size:0.75rem;">(Deadline: ${ai.deadline})</span>` : ''}
                </label>
            </div>
        `).join("");
    }

    // =========================================================================
    // Practice Quiz Carousel
    // =========================================================================
    let activeQuizPollInterval = null;

    async function triggerQuizRegeneration(btn) {
        if (!currentRecording?.id) return;
        const recId = currentRecording.id;
        const origLabel = btn ? btn.innerText : "Generate Exam Quizzes with LLM";
        if (btn) {
            btn.disabled = true;
            btn.innerText = "Starting Quiz Generation...";
        }
        if (activeQuizPollInterval) {
            clearInterval(activeQuizPollInterval);
            activeQuizPollInterval = null;
        }
        try {
            const res = await fetch(`/api/recordings/${recId}/intelligence/quizzes/regenerate`, {
                method: "POST"
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(data.detail || `Quiz regeneration failed (HTTP ${res.status})`);
            }

            activeQuizPollInterval = setInterval(async () => {
                try {
                    const pollRes = await fetch(`/api/recordings/${recId}/intelligence`);
                    if (!pollRes.ok) return;
                    const intel = await pollRes.json();
                    const qMeta = intel?.lecture_notes?.meta?.quiz_generation || {};
                    if (qMeta.status === "running") {
                        if ((intel.quizzes || []).length > 0) {
                            currentIntelligence = intel;
                            quizQuestions = intel.quizzes || [];
                            renderActiveQuizQuestion();
                        }
                        const curBtn = document.getElementById("retryQuizReprocessBtn") || document.getElementById("quizRegenerateBtn");
                        if (curBtn) {
                            curBtn.disabled = true;
                            curBtn.innerText = `Generating (${qMeta.generated || 0}/${qMeta.target || 10})...`;
                        }
                        return;
                    }

                    clearInterval(activeQuizPollInterval);
                    activeQuizPollInterval = null;
                    currentIntelligence = intel;
                    quizQuestions = intel.quizzes || [];
                    currentQuizIdx = 0;
                    renderActiveQuizQuestion();
                    if (quizQuestions.length > 0) {
                        showAlert(`Generated ${quizQuestions.length} exam-grade quiz questions.`, "success");
                    } else {
                        showAlert(qMeta.reason || "Quiz regeneration returned 0 questions.", "warning");
                    }
                } catch (_) {
                    // Keep polling on transient network hiccup
                }
            }, 4000);
        } catch (err) {
            showAlert(`Quiz regeneration failed: ${err.message}`, "error");
            if (btn) {
                btn.disabled = false;
                btn.innerText = origLabel;
            }
        }
    }

    function renderActiveQuizQuestion() {
        const wrapper = document.getElementById("interactiveQuizWrapper");
        let counter = document.getElementById("quizCounterDisplay");
        let qTitle = document.getElementById("quizQuestionText");
        let optionsGrid = document.getElementById("quizOptionsGrid");
        let answerCard = document.getElementById("quizAnswerCard");

        if (!quizQuestions || quizQuestions.length === 0) {
            const quizMeta = currentIntelligence?.lecture_notes?.meta?.quiz_generation;
            if (wrapper) {
                const isAudioProcessed = (currentRecording?.status || "").toUpperCase() === "COMPLETED";
                const isFailed = quizMeta && quizMeta.status === "failed";
                const isRunning = quizMeta && quizMeta.status === "running";
                const title = !isAudioProcessed
                    ? "Audio Processing Required"
                    : (isFailed
                        ? "Exam Quiz Generation Unavailable"
                        : (isRunning ? "Quiz Generation In Progress / Interrupted" : "No Exam Quizzes Generated Yet"));
                const rawMsg = !isAudioProcessed
                    ? "Complete audio processing first. Once processing finishes, you can generate exam-grade MCQs here if they are not generated."
                    : (isFailed
                        ? (quizMeta.reason || "Local LLM is unavailable or failed to produce valid exam questions.")
                        : (isRunning
                            ? `Progress: ${quizMeta.generated || 0}/${quizMeta.target || 10} questions. Click below to generate or resume.`
                            : "Click below to generate exam-grade MCQs for this lecture using the local LLM."));
                const msg = String(rawMsg).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                const boxStyle = isFailed && isAudioProcessed
                    ? "border:1px solid #FECACA; background:#FEF2F2;"
                    : "border:1px solid #E2E8F0; background:#F8FAFC;";
                const titleColor = isFailed && isAudioProcessed ? "#991B1B" : "#0F172A";
                const msgColor = isFailed && isAudioProcessed ? "#7F1D1D" : "#475569";

                wrapper.innerHTML = `
                    <div class="empty-state" style="padding:1.5rem; ${boxStyle} border-radius:10px; text-align:left;">
                        <div style="font-weight:700; color:${titleColor}; margin-bottom:6px;">${title}</div>
                        <div style="font-size:0.86rem; color:${msgColor}; margin-bottom:${isAudioProcessed ? "12px" : "0"};">${msg}</div>
                        ${isAudioProcessed ? `<button class="btn btn-primary btn-sm" id="retryQuizReprocessBtn">Generate Exam Quizzes with LLM</button>` : ""}
                    </div>
                `;
                if (isAudioProcessed) {
                    document.getElementById("retryQuizReprocessBtn")?.addEventListener("click", (e) => {
                        triggerQuizRegeneration(e.currentTarget);
                    });
                }
            }
            if (counter) counter.innerText = "0 / 0";
            return;
        }

        // If the wrapper previously had its template wiped by empty state, restore it!
        if (!qTitle || !optionsGrid) {
            if (wrapper) {
                wrapper.innerHTML = `
                    <div class="quiz-question-title" id="quizQuestionText">Loading quiz question...</div>
                    <div class="quiz-options-grid" id="quizOptionsGrid"></div>
                    <div class="quiz-answer-card hidden" id="quizAnswerCard">
                        <div class="quiz-answer-header">
                            <span id="quizCorrectAnswerLabel">Correct Answer</span>
                        </div>
                        <div class="quiz-explanation-text" id="quizExplanationText">Explanation details.</div>
                    </div>
                    <div class="quiz-actions-footer">
                        <button class="btn btn-secondary btn-sm" id="quizPrevBtn">Previous</button>
                        <button class="btn btn-secondary btn-sm" id="quizRevealBtn">Reveal Answer</button>
                        <button class="btn btn-secondary btn-sm" id="quizRegenerateBtn" title="Regenerate only the quiz section with LLM">↻ Regenerate Quizzes</button>
                        <button class="btn btn-primary btn-sm" id="quizNextBtn">Next Question →</button>
                    </div>
                `;
                // Re-bind listeners for next/prev/reveal buttons
                document.getElementById("quizPrevBtn")?.addEventListener("click", () => {
                    if (currentQuizIdx > 0) {
                        currentQuizIdx--;
                        renderActiveQuizQuestion();
                    }
                });
                document.getElementById("quizNextBtn")?.addEventListener("click", () => {
                    if (currentQuizIdx < quizQuestions.length - 1) {
                        currentQuizIdx++;
                        renderActiveQuizQuestion();
                    }
                });
                document.getElementById("quizRevealBtn")?.addEventListener("click", () => {
                    revealCurrentQuizAnswer();
                });
                document.getElementById("quizRegenerateBtn")?.addEventListener("click", (e) => {
                    triggerQuizRegeneration(e.currentTarget);
                });
            }
            qTitle = document.getElementById("quizQuestionText");
            optionsGrid = document.getElementById("quizOptionsGrid");
            answerCard = document.getElementById("quizAnswerCard");
        }

        if (currentQuizIdx >= quizQuestions.length) {
            currentQuizIdx = 0;
        }

        const q = quizQuestions[currentQuizIdx];
        const qMetaRunning = currentIntelligence?.lecture_notes?.meta?.quiz_generation;
        if (counter) {
            counter.innerText = (qMetaRunning && qMetaRunning.status === "running")
                ? `Question ${currentQuizIdx + 1} / ${quizQuestions.length} (Generating ${qMetaRunning.generated || quizQuestions.length}/${qMetaRunning.target || 10}...)`
                : `Question ${currentQuizIdx + 1} / ${quizQuestions.length}`;
        }

        if (qTitle && q) {
            const catHtml = q.category ? `<span style="background:#EEF2FF; color:#4F46E5; border:1px solid #C7D2FE; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:6px; display:inline-block;"><span class="mono-emoji indigo">🎯</span>${q.category}</span>` : "";
            const diffColors = {
                "Easy": "background:#ECFDF5; color:#047857; border:1px solid #A7F3D0;",
                "Medium": "background:#FFFBEB; color:#B45309; border:1px solid #FDE68A;",
                "Hard": "background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA;"
            };
            const diffStyle = diffColors[q.difficulty] || "background:#F1F5F9; color:#475569; border:1px solid #CBD5E1;";
            const diffHtml = q.difficulty ? `<span style="${diffStyle} font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:6px; display:inline-block;">${q.difficulty}</span>` : "";
            const topicHtml = q.topic_area ? `<span style="background:#F8FAFC; color:#475569; border:1px solid #E2E8F0; font-size:0.72rem; font-weight:600; padding:2px 8px; border-radius:6px; display:inline-block;">${q.topic_area}</span>` : "";
            qTitle.innerHTML = `
                <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;">${catHtml}${diffHtml}${topicHtml}</div>
                <div style="font-size:0.98rem; font-weight:700; color:#0F172A; line-height:1.45;">${currentQuizIdx + 1}. ${q.question || ""}</div>
            `;
        }

        if (answerCard) answerCard.classList.add("hidden");

        if (optionsGrid && q) {
            const letters = ["A", "B", "C", "D"];
            optionsGrid.innerHTML = (q.options || []).map((opt, optIdx) => {
                const letter = letters[optIdx] || `${optIdx + 1}`;
                return `
                    <div class="quiz-option-card" onclick="selectQuizOption(${optIdx})" style="display:flex; align-items:flex-start; gap:10px; padding:10px 14px; border-radius:8px; border:1.5px solid #E2E8F0; margin-bottom:6px; cursor:pointer; transition:all 0.15s; background:#ffffff;">
                        <input type="radio" name="quiz-radio" class="quiz-radio" id="opt-${optIdx}" style="margin-top:3px;">
                        <span style="font-weight:700; color:#4F46E5; min-width:18px;">${letter}.</span>
                        <span style="color:#1E293B; font-size:0.9rem; flex:1;">${opt}</span>
                    </div>
                `;
            }).join("");
        }
    }

    window.selectQuizOption = function(optIdx) {
        if (!quizQuestions || quizQuestions.length === 0) return;
        const q = quizQuestions[currentQuizIdx];

        document.querySelectorAll(".quiz-option-card").forEach((card, idx) => {
            card.classList.toggle("selected", idx === optIdx);
            const radio = card.querySelector(".quiz-radio");
            if (radio) radio.checked = (idx === optIdx);
        });

        const answerCard = document.getElementById("quizAnswerCard");
        const correctLabel = document.getElementById("quizCorrectAnswerLabel");
        const explanationText = document.getElementById("quizExplanationText");

        const ansLabel = q.correct_option_letter
            ? `<span class="mono-emoji emerald">✅</span>Correct Answer: ${q.correct_option_letter} (${q.correct_answer})`
            : `<span class="mono-emoji emerald">✅</span>Correct Answer: ${q.correct_answer}`;
        correctLabel.innerHTML = ansLabel;

        let explContent = q.explanation || "Grounded directly in lecture content.";
        if (q.calculation) {
            explContent += `\n\nStep-by-Step Calculation:\n${q.calculation}`;
        }
        explanationText.innerText = explContent;
        answerCard?.classList.remove("hidden");
    };

    function revealCurrentQuizAnswer() {
        if (!quizQuestions || quizQuestions.length === 0) return;
        const q = quizQuestions[currentQuizIdx];
        const answerCard = document.getElementById("quizAnswerCard");
        const correctLabel = document.getElementById("quizCorrectAnswerLabel");
        const explanationText = document.getElementById("quizExplanationText");

        const ansLabel = q.correct_option_letter
            ? `<span class="mono-emoji emerald">✅</span>Correct Answer: ${q.correct_option_letter} (${q.correct_answer})`
            : `<span class="mono-emoji emerald">✅</span>Correct Answer: ${q.correct_answer || 'Reference Answer'}`;
        correctLabel.innerHTML = ansLabel;

        let explContent = q.explanation || "Grounded directly in lecture concepts.";
        if (q.calculation) {
            explContent += `\n\nStep-by-Step Calculation:\n${q.calculation}`;
        }
        explanationText.innerText = explContent;
        answerCard?.classList.remove("hidden");
    }

    // =========================================================================
    // Transcripts Feed Renderer (Dual View & Filter)
    // =========================================================================
    function renderTranscripts(filterText = "") {
        const container = document.getElementById("transcriptSegmentsList");
        if (!currentSegments || currentSegments.length === 0) {
            container.innerHTML = `<div class="empty-state">No transcript segments available.</div>`;
            return;
        }

        const filtered = currentSegments.filter(seg => {
            if (!filterText) return true;
            const fullText = (seg.text || "") + " " + (seg.normalized_text || "");
            return fullText.toLowerCase().includes(filterText);
        });

        if (filtered.length === 0) {
            container.innerHTML = `<div class="empty-state">No spoken text matches "${filterText}".</div>`;
            return;
        }

        container.innerHTML = filtered.map(seg => {
            const startStr = formatDurationSimple(seg.start_time);
            const orig = seg.text || "";
            const norm = seg.normalized_text || orig;
            const english = seg.english_text || norm;
            const source = seg.translation_source || "passthrough";
            const conf = seg.translation_confidence !== undefined && seg.translation_confidence !== null ? Math.round(seg.translation_confidence * 100) : 100;

            let badgeHtml = "";
            if (source === "google_translate") {
                badgeHtml = `<span style="background:#E0E7FF; color:#4338CA; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:10px;" title="Translated via Google Translate API (Confidence: ${conf}%)">Google Translate</span>`;
            } else if (source === "local_llm") {
                badgeHtml = `<span style="background:#ECFDF5; color:#065F46; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:10px;" title="Translated via Local Qwen Model (Confidence: ${conf}%)">Local AI</span>`;
            } else if (source === "passthrough") {
                badgeHtml = `<span style="background:#F3F4F6; color:#374151; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:10px;" title="Native English detected (Confidence: 100%)">English</span>`;
            } else if (source === "dictionary") {
                badgeHtml = `<span style="background:#FEF3C7; color:#92400E; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:10px;" title="Dictionary fallback translation (Confidence: ${conf}%)">Dictionary</span>`;
            }

            const editBtnHtml = `
                <button class="btn-edit-segment" onclick="openEditSegmentModal('${seg.id || ''}', ${seg.segment_index})" title="Edit segment text and propagate downstream" style="background:#F1F5F9; border:1px solid #CBD5E1; border-radius:6px; font-size:0.72rem; color:#4F46E5; font-weight:600; padding:2px 8px; cursor:pointer; display:inline-flex; align-items:center; gap:3px; transition:all 0.15s;">
                    Edit
                </button>
            `;

            const segIdAttr = `data-segment-id="${seg.segment_index !== undefined ? seg.segment_index : (seg.id || '')}" data-segment-index="${seg.segment_index !== undefined ? seg.segment_index : ''}"`;

            if (currentTranscriptMode === "original") {
                return `
                    <div class="transcript-turn" data-start="${seg.start_time}" data-end="${seg.end_time}" ${segIdAttr}>
                        <button class="turn-timestamp" onclick="seekMedia(${seg.start_time})">${startStr}</button>
                        <div class="turn-content">
                            <div class="turn-speaker" style="display:flex; justify-content:space-between; align-items:center;">
                                <span>Lecturer <span style="font-weight:400; color:var(--text-subtle); font-size:0.7rem;">(Acoustic Spoken)</span></span>
                                ${editBtnHtml}
                            </div>
                            <div class="turn-text">${orig}</div>
                        </div>
                    </div>
                `;
            } else if (currentTranscriptMode === "normalized") {
                return `
                    <div class="transcript-turn" data-start="${seg.start_time}" data-end="${seg.end_time}" ${segIdAttr}>
                        <button class="turn-timestamp" onclick="seekMedia(${seg.start_time})">${startStr}</button>
                        <div class="turn-content">
                            <div class="turn-speaker" style="display:flex; justify-content:space-between; align-items:center;">
                                <span>Lecturer <span style="font-weight:400; color:var(--text-subtle); font-size:0.7rem;">(Cleaned Roman Urdu)</span></span>
                                ${editBtnHtml}
                            </div>
                            <div class="turn-text">${norm}</div>
                        </div>
                    </div>
                `;
            } else if (currentTranscriptMode === "english") {
                return `
                    <div class="transcript-turn" data-start="${seg.start_time}" data-end="${seg.end_time}" ${segIdAttr}>
                        <button class="turn-timestamp" onclick="seekMedia(${seg.start_time})">${startStr}</button>
                        <div class="turn-content">
                            <div class="turn-speaker" style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="display:inline-flex; align-items:center; gap:6px;">Lecturer <span style="font-weight:400; color:var(--text-subtle); font-size:0.7rem;">(English Translation)</span> ${badgeHtml}</span>
                                ${editBtnHtml}
                            </div>
                            <div class="turn-text" style="color:#1E293B; font-weight:500;">${english}</div>
                        </div>
                    </div>
                `;
            } else {
                // Tri-View (Side-by-Side: Spoken, Cleaned, English)
                return `
                    <div class="transcript-turn" data-start="${seg.start_time}" data-end="${seg.end_time}" ${segIdAttr}>
                        <button class="turn-timestamp" onclick="seekMedia(${seg.start_time})">${startStr}</button>
                        <div class="turn-content">
                            <div class="turn-speaker" style="display:flex; justify-content:space-between; align-items:center;">
                                <span>Lecturer</span>
                                ${editBtnHtml}
                            </div>
                            <div class="turn-dual-container">
                                <div class="dual-box">
                                    <div class="dual-label">Acoustic Spoken</div>
                                    <div>${orig}</div>
                                </div>
                                <div class="dual-box">
                                    <div class="dual-label" style="color:var(--primary);">Cleaned Roman Urdu</div>
                                    <div>${norm}</div>
                                </div>
                                <div class="dual-box" style="background:#F8FAFC; border-color:#CBD5E1;">
                                    <div class="dual-label" style="display:flex; justify-content:space-between; align-items:center; color:#0F766E;">
                                        <span>English Translation</span>
                                        ${badgeHtml}
                                    </div>
                                    <div style="color:#0F172A; font-weight:500;">${english}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
        }).join("");
    }

    // =========================================================================
    // Ask Anything — AI Q&A Chat
    // =========================================================================
    let activeQaSessionId = "sess_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);

    function refreshQAChat(recording, intel) {
        activeQaSessionId = "sess_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);
        const stream = document.getElementById("qaMessagesStream");
        const inputEl = document.getElementById("qaInput");
        if (inputEl) inputEl.value = "";
        if (!stream) return;

        const lectureTitle = recording && recording.filename
            ? recording.filename.replace(/\.[^/.]+$/, "")
            : "this lecture";

        stream.innerHTML = `
            <div class="chat-bubble ai-bubble">
                <div class="ai-msg-header"><span class="ai-avatar-pill">✨ AI Tutor</span></div>
                <div class="qa-chat-markdown">Assalamu alaikum! Ask any question about <strong>${escapeQuotes(lectureTitle)}</strong> or related concepts, or click a suggestion below.</div>
            </div>
        `;
        stream.scrollTop = 0;
        renderQASuggestions(intel);
    }

    function renderQASuggestions(intel) {
        const row = document.getElementById("qaSuggestionsRow");
        if (!row) return;
        const notes = (intel && intel.lecture_notes) || {};
        const topics = notes.topics || (notes.study_topics || []).map(t => t.title) || [];
        const topTopic = topics.length > 0 ? topics[0] : null;

        let html = `
            <button type="button" class="topic-tag-chip qa-suggestion-btn" data-q="What is the assignment topic and deadline?"><span class="mono-emoji">📋</span>Assignment deadline</button>
            <button type="button" class="topic-tag-chip qa-suggestion-btn" data-q="What are the key concepts discussed?"><span class="mono-emoji">📖</span>Key concepts</button>
            <button type="button" class="topic-tag-chip qa-suggestion-btn" data-q="Give me all the numerical formulas used in the lecture"><span class="mono-emoji">∑</span>Formulas</button>
            <button type="button" class="topic-tag-chip qa-suggestion-btn" data-q="What will come in the exam?"><span class="mono-emoji">🎯</span>Exam topics</button>
        `;
        if (topTopic) {
            html += `<button type="button" class="topic-tag-chip qa-suggestion-btn" data-q="Explain ${escapeQuotes(topTopic)} and its key details">${topTopic}</button>`;
        }
        row.innerHTML = html;
    }

    async function submitQuestion(query) {
        if (!currentRecording) return;
        const targetRecordingId = currentRecording.id;
        const stream = document.getElementById("qaMessagesStream");
        const inputEl = document.getElementById("qaInput");
        if (inputEl) inputEl.value = "";

        const userMsg = document.createElement("div");
        userMsg.className = "chat-bubble user-bubble";
        userMsg.innerText = query;
        stream.appendChild(userMsg);
        stream.scrollTop = stream.scrollHeight;

        const aiThinking = document.createElement("div");
        aiThinking.className = "chat-bubble ai-bubble";
        aiThinking.innerHTML = `<div class="ai-msg-header"><span class="ai-avatar-pill">✨ AI Tutor · GPU</span></div><div class="qa-chat-markdown" style="color:#64748B;">Thinking on GPU<span class="qa-streaming-cursor"></span></div>`;
        stream.appendChild(aiThinking);
        stream.scrollTop = stream.scrollHeight;

        const buildEvidenceHtml = (payload) => {
            if (Array.isArray(payload.evidence) && payload.evidence.length > 0) {
                return payload.evidence.slice(0, 2).map(e => {
                    const startSec = Number(e.start) || 0;
                    const segId = e.segment_id !== undefined && e.segment_id !== null ? JSON.stringify(e.segment_id) : 'null';
                    const timeLabel = formatDurationSimple(startSec);
                    const cleanText = (e.text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
                    const snippet = cleanText ? (cleanText.substring(0, 50) + (cleanText.length > 50 ? '...' : '')) : '';
                    const titleAttr = cleanText || `Topic starts at ${timeLabel}`;
                    return `<span class="evidence-chip" onclick="seekEvidence(${startSec}, ${segId})" title="${titleAttr}">` +
                        `<span class="evidence-chip-icon">⏱️</span> ` +
                        `<strong>${timeLabel}</strong>` +
                        (snippet ? `<span class="evidence-chip-text">: ${snippet}</span>` : '') +
                        `</span>`;
                }).join(" ");
            }
            if (payload.citations && payload.citations.length > 0) {
                return payload.citations.map(c => {
                    const seekSec = c.start_time !== undefined ? c.start_time : extractStartSeconds(c.timestamp);
                    const label = (c.timestamp || 'Jump to Timestamp').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                    return `<span class="evidence-chip" onclick="seekEvidence(${seekSec}, null)">` +
                        `<span class="evidence-chip-icon">⏱️</span> ` +
                        `<strong>${label}</strong>` +
                        `</span>`;
                }).join(" ");
            }
            if (payload.timestamp) {
                const seekSec = payload.start_time !== undefined ? payload.start_time : extractStartSeconds(payload.timestamp);
                const label = (payload.timestamp || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                return `<span class="evidence-chip" onclick="seekEvidence(${seekSec}, null)">` +
                    `<span class="evidence-chip-icon">⏱️</span> ` +
                    `<strong>${label}</strong>` +
                    `</span>`;
            }
            return "";
        };

        try {
            const isEli5 = document.getElementById("qaEli5Toggle")?.checked || false;
            const res = await fetch(`/api/recordings/${targetRecordingId}/qa/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: query,
                    top_k: 3,
                    eli5: isEli5,
                    session_id: activeQaSessionId
                })
            });

            if (!res.ok || !res.body) {
                throw new Error(`Streaming HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let sseBuf = "";
            let streamedText = "";
            let finalMeta = {};
            let lastRenderAt = 0;

            const renderStreamingBubble = (isDone = false) => {
                const bodyHtml = renderMarkdownToHtml(streamedText || "");
                const cursorHtml = isDone ? "" : `<span class="qa-streaming-cursor"></span>`;
                const evHtml = isDone ? buildEvidenceHtml(finalMeta) : "";
                const copyBtnHtml = isDone ? `<button type="button" class="qa-copy-btn">Copy</button>` : "";
                aiThinking.innerHTML = `
                    <div class="ai-msg-header"><span class="ai-avatar-pill">✨ AI Tutor · Qwen3.5</span>${copyBtnHtml}</div>
                    <div class="qa-chat-markdown">${bodyHtml}${cursorHtml}</div>
                    ${evHtml ? `<div class="qa-evidence-footer">${evHtml}</div>` : ''}
                `;
                if (isDone) {
                    const cpBtn = aiThinking.querySelector(".qa-copy-btn");
                    if (cpBtn) {
                        cpBtn.onclick = () => {
                            navigator.clipboard?.writeText(streamedText || "");
                            cpBtn.innerText = "Copied";
                            setTimeout(() => { cpBtn.innerText = "Copy"; }, 1400);
                        };
                    }
                }
                stream.scrollTop = stream.scrollHeight;
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                if (!currentRecording || currentRecording.id !== targetRecordingId) {
                    try { await reader.cancel(); } catch (_) {}
                    return;
                }
                sseBuf += decoder.decode(value, { stream: true });
                const blocks = sseBuf.split(/\r?\n\r?\n/);
                sseBuf = blocks.pop() || "";

                for (const block of blocks) {
                    const dataLine = block.split(/\r?\n/).find(l => l.startsWith("data:"));
                    if (!dataLine) continue;
                    let evt;
                    try {
                        evt = JSON.parse(dataLine.slice(5).trim());
                    } catch (_) {
                        continue;
                    }
                    if (evt.session_id) activeQaSessionId = evt.session_id;
                    if (evt.type === "meta") {
                        finalMeta = evt;
                    } else if (evt.type === "token" && evt.text) {
                        streamedText += evt.text;
                        const now = performance.now();
                        if (now - lastRenderAt >= 35 || evt.text.includes("\n")) {
                            renderStreamingBubble(false);
                            lastRenderAt = now;
                        }
                    } else if (evt.type === "done") {
                        finalMeta = Object.assign({}, finalMeta, evt);
                        renderStreamingBubble(true);
                    } else if (evt.type === "error") {
                        throw new Error(evt.message || "Streaming error");
                    }
                }
            }
            if (streamedText) {
                renderStreamingBubble(true);
            } else {
                aiThinking.innerText = "Could not retrieve answer from lecture.";
            }
        } catch (err) {
            try {
                const isEli5 = document.getElementById("qaEli5Toggle")?.checked || false;
                const res = await fetch(`/api/recordings/${targetRecordingId}/qa`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        question: query,
                        top_k: 3,
                        eli5: isEli5,
                        session_id: activeQaSessionId
                    })
                });
                const data = await res.json();
                if (!currentRecording || currentRecording.id !== targetRecordingId) return;
                if (data.session_id) activeQaSessionId = data.session_id;
                if (res.ok && data.status === "success") {
                    const formattedAnswer = renderMarkdownToHtml(data.answer || "No grounded answer available.");
                    const evidenceHtml = buildEvidenceHtml(data);
                    aiThinking.innerHTML = `
                        <div class="ai-msg-header"><span class="ai-avatar-pill">✨ AI Tutor · Qwen3.5</span></div>
                        <div class="qa-chat-markdown">${formattedAnswer}</div>
                        ${evidenceHtml ? `<div class="qa-evidence-footer">${evidenceHtml}</div>` : ''}
                    `;
                } else {
                    aiThinking.innerText = data.message || "Could not retrieve answer from lecture.";
                }
            } catch (fbErr) {
                if (!currentRecording || currentRecording.id !== targetRecordingId) return;
                aiThinking.innerText = `Error processing question: ${fbErr.message || err.message}`;
            }
        }
        stream.scrollTop = stream.scrollHeight;
    }

    // =========================================================================
    // Recordings List & History
    // =========================================================================
    let allRecordings = [];
    let activeLibraryFilter = "all";
    let librarySearchTerm = "";

    async function loadRecordings(autoOpenFirst = false) {
        try {
            const res = await fetch("/api/recordings");
            const data = await res.json();

            if (data.status === "success") {
                allRecordings = data.data || [];

                // Update Landing Hero Quick Stats
                const statTotalRecs = document.getElementById("statTotalRecordings");
                const statTotalHrs = document.getElementById("statTotalHours");
                if (statTotalRecs) statTotalRecs.innerText = allRecordings.length;
                if (statTotalHrs) {
                    const totalSeconds = allRecordings.reduce((acc, r) => acc + (r.duration_seconds || 0), 0);
                    statTotalHrs.innerText = (totalSeconds / 3600).toFixed(1) + "h";
                }

                applyLibraryFilters();

                if (autoOpenFirst && !currentRecording && allRecordings.length > 0) {
                    const preferred = allRecordings.find(r => r.filename.includes("operating")) || allRecordings.find(r => r.filename.includes("assign1")) || allRecordings.find(r => r.status === "COMPLETED") || allRecordings[0];
                    if (preferred) {
                        openDashboard(preferred.id);
                    }
                }
            }
        } catch (err) {
            if (recordingsList) recordingsList.innerHTML = `<div class="empty-state">Failed to load recordings: ${err.message}</div>`;
        }
    }

    function applyLibraryFilters() {
        if (!recordingsList) return;
        if (!allRecordings || allRecordings.length === 0) {
            recordingsList.innerHTML = `<div class="empty-state">No recordings registered yet. Upload an audio or video file above.</div>`;
            return;
        }

        let filtered = allRecordings.filter(rec => {
            // Search filter
            if (librarySearchTerm) {
                const name = (rec.filename || "").toLowerCase();
                if (!name.includes(librarySearchTerm)) return false;
            }

            // Category filter
            if (activeLibraryFilter === "completed") {
                return rec.status === "COMPLETED";
            } else if (activeLibraryFilter === "video") {
                return rec.media_type === "video" || /\.(mp4|webm|mov)$/i.test(rec.filename || "");
            } else if (activeLibraryFilter === "audio") {
                return rec.media_type === "audio" || /\.(mp3|wav|m4a|ogg|flac)$/i.test(rec.filename || "");
            }
            return true; // 'all'
        });

        if (filtered.length === 0) {
            recordingsList.innerHTML = `<div class="empty-state">No saved recordings match the selected filter or search query.</div>`;
            return;
        }

        renderRecordingsList(filtered);
    }

    function renderRecordingsList(list) {
        if (!recordingsList) return;
        recordingsList.innerHTML = list.map(rec => {
            const isCompleted = rec.status === "COMPLETED";
            const isFailed = rec.status === "FAILED";
            const statusClass = isCompleted ? "badge-success" : (isFailed ? "badge-danger" : "badge-warning");
            const isVideo = rec.media_type === "video" || /\.(mp4|webm|mov)$/i.test(rec.filename || "");

            const mediaIconSvg = isVideo
                ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>`
                : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>`;

            return `
                <div class="recording-item">
                    <div class="recording-info">
                        <div class="media-icon-box" title="${isVideo ? 'Video Recording' : 'Audio Recording'}" style="width:36px; height:36px; border-radius:8px; background:${isVideo ? '#EEF2FF' : '#F0FDF4'}; color:${isVideo ? '#4F46E5' : '#166534'}; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                            ${mediaIconSvg}
                        </div>
                        <div>
                            <div class="recording-title" style="font-weight:700; color:var(--text-main); font-size:0.92rem;">${rec.filename}</div>
                            <div class="recording-sub" style="display:flex; align-items:center; gap:6px; flex-wrap:wrap; font-size:0.75rem; color:var(--text-muted); margin-top:2px;">
                                <span class="status-badge-chip ${statusClass}">${rec.status}</span>
                                <span>•</span>
                                <span>${formatBytes(rec.file_size_bytes)}</span>
                                <span>•</span>
                                <span>${formatDurationSimple(rec.duration_seconds)}</span>
                                <span>•</span>
                                <span>${new Date(rec.created_at || Date.now()).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}</span>
                            </div>
                        </div>
                    </div>
                    <div class="recording-actions" style="display:flex; align-items:center; gap:6px;">
                        <button class="btn btn-primary btn-sm" onclick="window.openDashboard('${rec.id}')" title="Open workspace and view preserved notes, quizzes, and transcript">
                            <span>Open Workspace</span>
                        </button>
                        ${!isCompleted ? `<button class="btn btn-success btn-sm" onclick="startProcessing('${rec.id}')" title="Process recording">Process</button>` : `<button class="btn btn-secondary btn-sm" onclick="openReprocessModal('${rec.id}', '${escapeQuotes(rec.filename)}')" title="Re-process options">Re-process</button>`}
                        <button class="btn btn-danger btn-sm" onclick="deleteRecording('${rec.id}')" title="Delete recording">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                        </button>
                    </div>
                </div>
            `;
        }).join("");
    }

    window.deleteRecording = async function(id) {
        if (!confirm(`Are you sure you want to delete recording ${id}?`)) return;
        try {
            const res = await fetch(`/api/recordings/${id}`, { method: "DELETE" });
            const data = await res.json();
            if (res.ok) {
                showAlert("Recording deleted successfully.", "success");
                loadRecordings(false);
            } else {
                showAlert(data.message || "Failed to delete recording.", "error");
            }
        } catch (err) {
            showAlert(`Delete failed: ${err.message}`, "error");
        }
    };

    // =========================================================================
    // Modals Management
    // =========================================================================
    function openExportModal() {
        const modal = document.getElementById("exportModal");
        if (modal) modal.classList.remove("hidden");
    }

    function closeExportModal() {
        const modal = document.getElementById("exportModal");
        if (modal) modal.classList.add("hidden");
    }

    function triggerExport(format) {
        if (!currentRecording) {
            showAlert("Please select a recording to export.", "info");
            return;
        }
        window.open(`/api/recordings/${currentRecording.id}/export?content_type=full&format=${format}`, "_blank");
        closeExportModal();
    }

    function openDetailsModal(title, bodyHtml) {
        const modal = document.getElementById("detailsModal");
        const titleEl = document.getElementById("detailsModalTitle");
        const bodyEl = document.getElementById("detailsModalBody");
        if (titleEl) titleEl.innerText = title;
        if (bodyEl) bodyEl.innerHTML = bodyHtml;
        if (modal) modal.classList.remove("hidden");
    }

    function closeDetailsModal() {
        const modal = document.getElementById("detailsModal");
        if (modal) modal.classList.add("hidden");
    }

    // =========================================================================
    // Manual Segment Correction & Downstream Recomputation
    // =========================================================================
    let currentEditingSegment = null;

    window.openEditSegmentModal = function(segmentId, segmentIndex) {
        if (!currentSegments || currentSegments.length === 0) return;

        let seg = currentSegments.find(s => s.id === segmentId);
        if (!seg) {
            seg = currentSegments.find(s => s.segment_index === segmentIndex);
        }
        if (!seg) return;

        currentEditingSegment = seg;

        const badge = document.getElementById("editSegmentBadge");
        const subtitle = document.getElementById("editSegmentSubtitle");
        const normInput = document.getElementById("editSegmentNormText");
        const rawInput = document.getElementById("editSegmentRawText");
        const statusMsg = document.getElementById("editSegmentStatusMsg");
        const saveBtn = document.getElementById("saveEditSegmentBtn");

        if (badge) badge.innerText = `Seg #${seg.segment_index}`;
        if (subtitle) {
            subtitle.innerText = `Timestamp: ${formatDurationSimple(seg.start_time)} - ${formatDurationSimple(seg.end_time)} | Chunk #${seg.chunk_index}`;
        }
        if (normInput) normInput.value = seg.normalized_text || seg.text || "";
        if (rawInput) rawInput.value = seg.text || "";
        if (statusMsg) {
            statusMsg.className = "hidden";
            statusMsg.innerHTML = "";
        }
        if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.innerHTML = "<span>Save & Propagate</span>";
        }

        const fullRadio = document.querySelector('input[name="editRecomputeScope"][value="full"]');
        if (fullRadio) fullRadio.checked = true;

        const modal = document.getElementById("editSegmentModal");
        if (modal) modal.classList.remove("hidden");
    };

    window.closeEditSegmentModal = function() {
        const modal = document.getElementById("editSegmentModal");
        if (modal) modal.classList.add("hidden");
        currentEditingSegment = null;
    };

    document.getElementById("closeEditSegmentModalBtn")?.addEventListener("click", window.closeEditSegmentModal);
    document.getElementById("cancelEditSegmentBtn")?.addEventListener("click", window.closeEditSegmentModal);

    document.getElementById("saveEditSegmentBtn")?.addEventListener("click", async () => {
        if (!currentRecording || !currentEditingSegment) return;

        const normText = document.getElementById("editSegmentNormText")?.value?.trim() || "";
        const rawText = document.getElementById("editSegmentRawText")?.value?.trim() || "";
        const scope = document.querySelector('input[name="editRecomputeScope"]:checked')?.value || "full";

        const saveBtn = document.getElementById("saveEditSegmentBtn");
        const statusMsg = document.getElementById("editSegmentStatusMsg");

        if (saveBtn) saveBtn.disabled = true;
        if (statusMsg) {
            statusMsg.className = "";
            statusMsg.style.background = "#EEF2FF";
            statusMsg.style.color = "#3730A3";
            statusMsg.innerHTML = `
                <div style="width:14px; height:14px; border:2px solid #4F46E5; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite;"></div>
                <span>Saving correction & propagating downstream (${scope === 'full' ? 'Full recompute: summary, notes, assignments, search embeddings' : 'Fast chunk recompute'})...</span>
            `;
        }

        try {
            const resp = await fetch(`/api/recordings/${currentRecording.id}/segments/${currentEditingSegment.id}/correct`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    normalized_text: normText,
                    text: rawText,
                    scope: scope
                })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Failed to update segment.");
            }

            const data = await resp.json();

            // 1. Update in-memory segment
            currentEditingSegment.normalized_text = normText;
            currentEditingSegment.text = rawText;

            // 2. Re-render transcript feed
            renderTranscripts();

            // 3. Live update downstream cards if returned
            if (data.intelligence) {
                currentIntelligence = data.intelligence;
                renderDashboard(false);
            }

            showAlert(`Segment #${currentEditingSegment.segment_index} corrected! Downstream embeddings & intelligence successfully updated.`, "success");
            window.closeEditSegmentModal();
        } catch (err) {
            if (statusMsg) {
                statusMsg.className = "";
                statusMsg.style.background = "#FEF2F2";
                statusMsg.style.color = "#991B1B";
                statusMsg.innerHTML = `<span>${err.message}</span>`;
            }
            if (saveBtn) saveBtn.disabled = false;
        }
    });

    function escapeQuotes(str) {
        return (str || "").replace(/'/g, "\\'").replace(/"/g, '&quot;');
    }
});
