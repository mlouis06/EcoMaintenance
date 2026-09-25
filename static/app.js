const questionBox =
    document.getElementById("question");

const sendButton =
    document.getElementById("send-button");

const conversation =
    document.getElementById("conversation");

const welcomeState =
    document.getElementById("welcome-state");

const chatScroll =
    document.getElementById("chat-scroll");


const sourcesPanel =
    document.getElementById("sources-panel");

const sourcesToggle =
    document.getElementById("sources-toggle");

const composerSourcesButton =
    document.getElementById("composer-sources-button");

const drawerClose =
    document.getElementById("drawer-close");

const drawerBackdrop =
    document.getElementById("drawer-backdrop");


const manualList =
    document.getElementById("manual-list");

const drawingList =
    document.getElementById("drawing-list");

const manualCount =
    document.getElementById("manual-count");

const drawingCount =
    document.getElementById("drawing-count");

const refreshDocumentsButton =
    document.getElementById("refresh-documents");


const uploadManualButton =
    document.getElementById("upload-manual-button");

const uploadDrawingButton =
    document.getElementById("upload-drawing-button");

const manualFileInput =
    document.getElementById("manual-file-input");

const drawingFileInput =
    document.getElementById("drawing-file-input");

const uploadStatus =
    document.getElementById("upload-status");


const attachmentBar =
    document.getElementById("attachment-bar");

const attachmentList =
    document.getElementById("attachment-list");

const clearDrawingsButton =
    document.getElementById("clear-drawings");


let selectedDrawings = [];

let cachedManuals = [];

let cachedDrawings = [];

let isSending = false;


/* ==========================================================
   SMALL HELPERS
   ========================================================== */

function escapeHtml(value) {

    return String(value ?? "")

        .replaceAll("&", "&amp;")

        .replaceAll("<", "&lt;")

        .replaceAll(">", "&gt;")

        .replaceAll('"', "&quot;")

        .replaceAll("'", "&#039;");
}


function hideWelcome() {

    welcomeState.style.display =
        "none";
}


function scrollToBottom() {

    window.requestAnimationFrame(
        () => {

            chatScroll.scrollTo({

                top:
                    chatScroll.scrollHeight,

                behavior:
                    "smooth"
            });

        }
    );
}


function isMobileLayout() {

    return window
        .matchMedia(
            "(max-width: 900px)"
        )
        .matches;
}


function openSourcesDrawer() {

    if (!isMobileLayout()) {
        return;
    }


    sourcesPanel.classList.add(
        "open"
    );


    drawerBackdrop.classList.add(
        "open"
    );


    sourcesToggle.setAttribute(
        "aria-expanded",
        "true"
    );
}


function closeSourcesDrawer() {

    sourcesPanel.classList.remove(
        "open"
    );


    drawerBackdrop.classList.remove(
        "open"
    );


    sourcesToggle.setAttribute(
        "aria-expanded",
        "false"
    );
}


function autosizeQuestionBox() {

    questionBox.style.height =
        "auto";


    questionBox.style.height =
        `${Math.min(
            questionBox.scrollHeight,
            180
        )}px`;
}


function showUploadMessage(
    message,
    isError = false
) {

    uploadStatus.textContent =
        message;


    uploadStatus.classList.toggle(
        "error",
        isError
    );
}


function sourceTypeLabel(source) {

    if (
        source.type ===
        "local_manual"
    ) {
        return "Manual";
    }


    if (
        source.type ===
        "trusted_web"
    ) {
        return "Manufacturer source";
    }


    if (
        source.type ===
        "uploaded_visual"
    ) {
        return "Drawing / image";
    }


    return "Source";
}


function selectedDrawingIds() {

    return selectedDrawings.map(
        item => item.document_id
    );
}


/* ==========================================================
   ANSWER FORMATTING

   We do not trust model generated HTML.
   The backend sends text and we build the UI ourselves.
   ========================================================== */

function appendInlineFormatting(
    parent,
    text
) {

    const pattern =
        /(\*\*[^*]+\*\*|\[Sources?[^\]]+\])/gi;


    let lastIndex = 0;

    let match;


    while (
        (
            match =
                pattern.exec(text)
        ) !== null
    ) {

        if (
            match.index >
            lastIndex
        ) {

            parent.appendChild(

                document.createTextNode(

                    text.slice(
                        lastIndex,
                        match.index
                    )
                )
            );
        }


        const token =
            match[0];


        if (
            token.startsWith("**")
        ) {

            const strong =
                document.createElement(
                    "strong"
                );


            strong.textContent =
                token.slice(
                    2,
                    -2
                );


            parent.appendChild(
                strong
            );

        } else {

            const citation =
                document.createElement(
                    "span"
                );


            citation.className =
                "citation";


            citation.textContent =
                token;


            parent.appendChild(
                citation
            );
        }


        lastIndex =
            pattern.lastIndex;
    }


    if (
        lastIndex <
        text.length
    ) {

        parent.appendChild(

            document.createTextNode(

                text.slice(
                    lastIndex
                )
            )
        );
    }
}


function renderAnswerText(
    container,
    rawText
) {

    const text =
        String(
            rawText || ""
        ).trim();


    if (!text) {

        const p =
            document.createElement(
                "p"
            );


        p.textContent =
            "No answer was returned.";


        container.appendChild(
            p
        );


        return;
    }


    /*
       sometimes a model puts several
       bullet points on one line.

       this cleans that up before rendering.
    */

    const normalized =
        text

            .replace(
                /\s+-\s+(?=[A-Z][^:]{1,80}:)/g,
                "\n- "
            )

            .replace(
                /\s+(Safety:)/gi,
                "\n\n$1"
            );


    const lines =
        normalized

            .split(/\n+/)

            .map(
                line =>
                    line.trim()
            )

            .filter(Boolean);


    let list = null;


    for (
        const line of lines
    ) {

        const bulletMatch =
            line.match(
                /^[-•]\s+(.*)$/
            );


        const numberedMatch =
            line.match(
                /^\d+[.)]\s+(.*)$/
            );


        if (
            bulletMatch ||
            numberedMatch
        ) {

            if (!list) {

                list =
                    document.createElement(

                        numberedMatch
                            ? "ol"
                            : "ul"
                    );


                container.appendChild(
                    list
                );
            }


            const li =
                document.createElement(
                    "li"
                );


            appendInlineFormatting(

                li,

                (
                    bulletMatch ||
                    numberedMatch
                )[1]
            );


            list.appendChild(
                li
            );


            continue;
        }


        list = null;


        const p =
            document.createElement(
                "p"
            );


        appendInlineFormatting(
            p,
            line
        );


        container.appendChild(
            p
        );
    }
}


/* ==========================================================
   SOURCE / EVIDENCE CARDS
   ========================================================== */

function createSourceSection(
    sources
) {

    if (
        !Array.isArray(sources) ||
        sources.length === 0
    ) {

        return null;
    }


    const section =
        document.createElement(
            "section"
        );


    section.className =
        "source-section";


    const heading =
        document.createElement(
            "h4"
        );


    heading.textContent =
        "Evidence";


    section.appendChild(
        heading
    );


    const grid =
        document.createElement(
            "div"
        );


    grid.className =
        "source-grid";


    sources.forEach(
        (
            source,
            index
        ) => {

            const details =
                document.createElement(
                    "details"
                );


            details.className =
                "source-card";


            const summary =
                document.createElement(
                    "summary"
                );


            const head =
                document.createElement(
                    "div"
                );


            head.className =
                "source-head";


            const left =
                document.createElement(
                    "div"
                );


            const sourceIndex =
                document.createElement(
                    "span"
                );


            sourceIndex.className =
                "source-index";


            sourceIndex.textContent =
                `SOURCE ${
                    source.source_number ??
                    index + 1
                }`;


            const title =
                document.createElement(
                    "span"
                );


            title.className =
                "source-title";


            title.textContent =
                source.title ||
                "Technical source";


            const meta =
                document.createElement(
                    "div"
                );


            meta.className =
                "source-meta";


            const metaParts = [

                sourceTypeLabel(
                    source
                )
            ];


            if (source.page) {

                metaParts.push(
                    `Page ${source.page}`
                );
            }


            if (
                Array.isArray(
                    source.pages_analyzed
                ) &&
                source.pages_analyzed.length
            ) {

                metaParts.push(

                    `Pages ${
                        source
                            .pages_analyzed
                            .join(", ")
                    }`
                );
            }


            if (source.ocr_used) {

                metaParts.push(
                    "OCR"
                );
            }


            meta.textContent =
                metaParts.join(
                    " · "
                );


            left.appendChild(
                sourceIndex
            );


            left.appendChild(
                title
            );


            left.appendChild(
                meta
            );


            head.appendChild(
                left
            );


            /*
               Local manuals and web sources
               may both have a URL.
            */

            if (source.url) {

                const link =
                    document.createElement(
                        "a"
                    );


                link.className =
                    "source-open";


                link.href =
                    source.url;


                link.target =
                    "_blank";


                link.rel =
                    "noopener noreferrer";


                link.textContent =
                    "Open ↗";


                link.addEventListener(

                    "click",

                    event =>
                        event.stopPropagation()
                );


                head.appendChild(
                    link
                );
            }


            summary.appendChild(
                head
            );


            details.appendChild(
                summary
            );


            /*
               Show the exact evidence the
               answer was based on.
            */

            if (source.excerpt) {

                const excerpt =
                    document.createElement(
                        "p"
                    );


                excerpt.className =
                    "source-excerpt";


                excerpt.textContent =
                    source.excerpt;


                details.appendChild(
                    excerpt
                );
            }


            grid.appendChild(
                details
            );
        }
    );


    section.appendChild(
        grid
    );


    return section;
}


/* ==========================================================
   CONVERSATION
   ========================================================== */

function addUserMessage(
    text
) {

    hideWelcome();


    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message message-user";


    const bubble =
        document.createElement(
            "div"
        );


    bubble.className =
        "user-bubble";


    bubble.textContent =
        text;


    wrapper.appendChild(
        bubble
    );


    conversation.appendChild(
        wrapper
    );


    scrollToBottom();
}


function addLoadingMessage() {

    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message";


    wrapper.dataset.loading =
        "true";


    wrapper.innerHTML = `
        <div class="loading-card">
            <span class="spinner"></span>
            <span>
                Reviewing your technical sources…
            </span>
        </div>
    `;


    conversation.appendChild(
        wrapper
    );


    scrollToBottom();


    return wrapper;
}


function addAssistantMessage(
    data
) {

    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message";


    const card =
        document.createElement(
            "article"
        );


    card.className =
        "answer-card";


    const header =
        document.createElement(
            "div"
        );


    header.className =
        "answer-header";


    const headingBlock =
        document.createElement(
            "div"
        );


    headingBlock.innerHTML = `
        <div class="answer-kicker">
            EcoMaintenance
        </div>

        <h3 class="answer-title">
            Troubleshooting guidance
        </h3>
    `;


    const sourceCount =
        document.createElement(
            "span"
        );


    sourceCount.className =
        "source-count";


    const count =
        Array.isArray(
            data.sources
        )
            ? data.sources.length
            : 0;


    sourceCount.textContent =
        count === 1
            ? "1 source"
            : `${count} sources`;


    header.appendChild(
        headingBlock
    );


    header.appendChild(
        sourceCount
    );


    card.appendChild(
        header
    );


    const body =
        document.createElement(
            "div"
        );


    body.className =
        "answer-body";


    renderAnswerText(
        body,
        data.answer
    );


    card.appendChild(
        body
    );


    const sourceSection =
        createSourceSection(
            data.sources || []
        );


    if (sourceSection) {

        card.appendChild(
            sourceSection
        );
    }


    wrapper.appendChild(
        card
    );


    conversation.appendChild(
        wrapper
    );


    scrollToBottom();
}


function addErrorMessage(
    message
) {

    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message";


    const card =
        document.createElement(
            "div"
        );


    card.className =
        "error-card";


    card.textContent =
        message;


    wrapper.appendChild(
        card
    );


    conversation.appendChild(
        wrapper
    );


    scrollToBottom();
}


/* ==========================================================
   DOCUMENTS
   ========================================================== */

async function loadDocuments() {

    try {

        const response =
            await fetch(
                "/documents"
            );


        if (!response.ok) {

            throw new Error(
                "Could not load sources"
            );
        }


        const data =
            await response.json();


        cachedManuals =
            data.manuals || [];


        cachedDrawings =
            data.drawings || [];


        /*
           If a visual gets removed from the
           backend, do not keep it attached.
        */

        const validDrawingIds =
            new Set(

                cachedDrawings.map(
                    item =>
                        item.document_id
                )
            );


        selectedDrawings =
            selectedDrawings.filter(

                item =>
                    validDrawingIds.has(
                        item.document_id
                    )
            );


        renderDocuments();

        renderAttachments();

    } catch (error) {

        console.error(
            error
        );


        showUploadMessage(
            "Could not reach the backend.",
            true
        );
    }
}


function renderDocuments() {

    manualList.innerHTML =
        "";


    drawingList.innerHTML =
        "";


    manualCount.textContent =
        cachedManuals.length;


    drawingCount.textContent =
        cachedDrawings.length;


    /*
       Every manual is searched
       automatically.
    */

    if (
        cachedManuals.length === 0
    ) {

        manualList.innerHTML = `
            <div class="empty-documents">
                No manuals uploaded yet
            </div>
        `;

    } else {

        cachedManuals.forEach(
            manual => {

                const item =
                    document.createElement(
                        "div"
                    );


                item.className =
                    "document-item";


                item.innerHTML = `
                    <span class="document-name">
                        ${escapeHtml(manual.name)}
                    </span>

                    <span class="document-meta">
                        Searched automatically
                    </span>
                `;


                manualList.appendChild(
                    item
                );
            }
        );
    }


    /*
       Drawings are optional context.

       The user chooses which ones should
       be considered with the next question.
    */

    if (
        cachedDrawings.length === 0
    ) {

        drawingList.innerHTML = `
            <div class="empty-documents">
                No drawings or images yet
            </div>
        `;

    } else {

        cachedDrawings.forEach(
            drawing => {

                const button =
                    document.createElement(
                        "button"
                    );


                button.type =
                    "button";


                button.className =
                    "document-item selectable";


                button.innerHTML = `
                    <span class="document-name">
                        ${escapeHtml(drawing.name)}
                    </span>

                    <span class="document-meta">
                        Click to include / remove
                    </span>
                `;


                if (
                    selectedDrawings.some(

                        item =>
                            item.document_id ===
                            drawing.document_id
                    )
                ) {

                    button.classList.add(
                        "selected"
                    );
                }


                button.addEventListener(

                    "click",

                    () =>
                        toggleDrawing(
                            drawing
                        )
                );


                drawingList.appendChild(
                    button
                );
            }
        );
    }
}


function toggleDrawing(
    drawing
) {

    const exists =
        selectedDrawings.some(

            item =>
                item.document_id ===
                drawing.document_id
        );


    if (exists) {

        selectedDrawings =
            selectedDrawings.filter(

                item =>
                    item.document_id !==
                    drawing.document_id
            );

    } else {

        /*
           Keeping this limited prevents one
           question from sending huge amounts
           of visual context.
        */

        if (
            selectedDrawings.length >= 3
        ) {

            showUploadMessage(

                "You can include up to 3 drawings/images in one question.",

                true
            );


            return;
        }


        selectedDrawings.push(
            drawing
        );
    }


    renderDocuments();

    renderAttachments();


    if (
        isMobileLayout()
    ) {

        closeSourcesDrawer();
    }
}


function renderAttachments() {

    attachmentList.innerHTML =
        "";


    if (
        selectedDrawings.length === 0
    ) {

        attachmentBar.classList.add(
            "hidden"
        );


        return;
    }


    attachmentBar.classList.remove(
        "hidden"
    );


    selectedDrawings.forEach(
        drawing => {

            const chip =
                document.createElement(
                    "span"
                );


            chip.className =
                "attachment-chip";


            const label =
                document.createElement(
                    "span"
                );


            label.textContent =
                drawing.name;


            const remove =
                document.createElement(
                    "button"
                );


            remove.type =
                "button";


            remove.textContent =
                "×";


            remove.setAttribute(

                "aria-label",

                `Remove ${drawing.name}`
            );


            remove.addEventListener(

                "click",

                () =>
                    toggleDrawing(
                        drawing
                    )
            );


            chip.appendChild(
                label
            );


            chip.appendChild(
                remove
            );


            attachmentList.appendChild(
                chip
            );
        }
    );
}


/* ==========================================================
   MULTIPLE FILE UPLOAD
   ========================================================== */

async function uploadMany(
    files,
    endpoint,
    label
) {

    const chosen =
        Array.from(
            files || []
        );


    if (
        chosen.length === 0
    ) {

        return;
    }


    const form =
        new FormData();


    /*
       FastAPI gets these as:
       list[UploadFile]
    */

    chosen.forEach(
        file => {

            form.append(
                "files",
                file
            );
        }
    );


    showUploadMessage(

        `Uploading ${
            chosen.length
        } ${
            label
        }${
            chosen.length === 1
                ? ""
                : "s"
        }…`
    );


    try {

        const response =
            await fetch(
                endpoint,
                {
                    method:
                        "POST",

                    body:
                        form
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(

                data.detail ||

                `Could not upload ${label}`
            );
        }


        const uploaded =
            data.files || [];


        const warnings =
            uploaded

                .map(
                    item =>
                        item.warning
                )

                .filter(Boolean);


        showUploadMessage(

            warnings.length

                ? `${
                    uploaded.length
                } uploaded. ${
                    warnings[0]
                }`

                : `${
                    uploaded.length
                } ${
                    label
                }${
                    uploaded.length === 1
                        ? ""
                        : "s"
                } uploaded.`
        );


        await loadDocuments();

    } catch (error) {

        console.error(
            error
        );


        showUploadMessage(

            error.message ||
            "Upload failed.",

            true
        );
    }
}


/* ==========================================================
   ASK BACKEND
   ========================================================== */

async function submitQuestion() {

    if (isSending) {
        return;
    }


    const question =
        String(
            questionBox.value
        ).trim();


    if (!question) {
        return;
    }


    isSending =
        true;


    sendButton.disabled =
        true;


    addUserMessage(
        question
    );


    const loadingMessage =
        addLoadingMessage();


    questionBox.value =
        "";


    autosizeQuestionBox();


    try {

        /*
           All manuals are already searched
           by the backend.

           drawing_ids only tells the backend
           which visual files should be added
           to this specific question.
        */

        const payload = {

            question:

                question,


            drawing_ids:

                selectedDrawingIds()
        };


        const response =
            await fetch(
                "/ask",
                {
                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            payload
                        )
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(

                data.detail ||

                "EcoMaintenance could not complete the request."
            );
        }


        loadingMessage.remove();


        addAssistantMessage(
            data
        );

    } catch (error) {

        console.error(
            error
        );


        loadingMessage.remove();


        addErrorMessage(

            error.message ||

            "EcoMaintenance could not complete the request."
        );

    } finally {

        isSending =
            false;


        sendButton.disabled =
            false;


        questionBox.focus();
    }
}


/* ==========================================================
   EVENTS
   ========================================================== */

sendButton.addEventListener(
    "click",
    submitQuestion
);


questionBox.addEventListener(

    "input",

    autosizeQuestionBox
);


questionBox.addEventListener(

    "keydown",

    event => {

        /*
           Enter sends.
           Shift + Enter makes a new line.
        */

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            submitQuestion();
        }
    }
);


uploadManualButton.addEventListener(

    "click",

    () =>
        manualFileInput.click()
);


uploadDrawingButton.addEventListener(

    "click",

    () =>
        drawingFileInput.click()
);


manualFileInput.addEventListener(

    "change",

    async () => {

        await uploadMany(

            manualFileInput.files,

            "/upload/manuals",

            "manual"
        );


        manualFileInput.value =
            "";
    }
);


drawingFileInput.addEventListener(

    "change",

    async () => {

        await uploadMany(

            drawingFileInput.files,

            "/upload/drawings",

            "visual"
        );


        drawingFileInput.value =
            "";
    }
);


clearDrawingsButton.addEventListener(

    "click",

    () => {

        selectedDrawings =
            [];


        renderDocuments();

        renderAttachments();
    }
);


sourcesToggle.addEventListener(

    "click",

    () => {

        if (
            sourcesPanel.classList.contains(
                "open"
            )
        ) {

            closeSourcesDrawer();

        } else {

            openSourcesDrawer();
        }
    }
);


composerSourcesButton.addEventListener(

    "click",

    openSourcesDrawer
);


drawerClose.addEventListener(

    "click",

    closeSourcesDrawer
);


drawerBackdrop.addEventListener(

    "click",

    closeSourcesDrawer
);


window.addEventListener(

    "resize",

    () => {

        if (
            !isMobileLayout()
        ) {

            closeSourcesDrawer();
        }
    }
);


refreshDocumentsButton.addEventListener(

    "click",

    loadDocuments
);


/* ==========================================================
   START APP
   ========================================================== */

loadDocuments();

autosizeQuestionBox();

questionBox.focus();