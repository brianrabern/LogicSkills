// ---- Bundles ----
// Each bundle file must be a small JSON object with shape:
// { countermodel: [{id,text}], symbolization: [{id,text},{id,text}], validity: [{id,text},{id,text}] }
const BUNDLE_FILE_IDS = [
  "12jPeSwyfaJqTZjYnsc7oGFZ4ObK3Xvyh",
  "1DsIiP3BJ9czcH8XRIHg7QoZp7f2MTDOx",
  "1_aoY-kN8hD5oYjwvg3K8DwWnYkc_8bkF",
  "1qdwhKFhV3hEntlgJuIr-5Seiu-S1O6en",
  "1yfMCKo4W1qJfEHCekqgSE-f5UWCNeOGC",
  "12LP9EsHIryJIKt12XPqa453AcDsdvHXZ",
  "1QgL5nbYOnckWYHM0ygJDnAv-djtdDv1q",
  "1oFbaH_2KyPSvk0YCwokP5o4A4TCvjrXM",
  "1RHH8kPuJ_z75O9e-MvXz-YtfJ3tfV8_Q",
  "1wKxVeOgpE_HvUoEOTGqwe1lKYEgjVmTd",
  "1qeTTbs-4509qdQRKj5Prlx3EWpjdoXHN",
  "1Ue_YersEYb5qy39-NV3k3yEbh9lg_s2Y",
  "1Y_nfwHzgdc8M9wj5fS40J4le50pygr5U",
  "17q07cORdk54ny7Xgth6ENt2SLjH9h5Jv",
  "1efbwZFzW0VJzJREZC61rEu-7JGqn3kjP",
  "1CKxm2_mDsX15gyPqy_O5FvFZpOWKggQx",
  "1L0lTxsdX_nnlAn4ijepnjhwvKhgyUjoe",
  "1DNPBd1Lz8_sf3g49QuhuJlcCMbyaUZLw",
  "1HM7-Tck8fAlcAyViutzG0S0UuiF8QIdj",
  "1jn5EgY1JMlf1sUnjzEXl12f4xkqm9iUh",
  "1ixeuSo2KRO9rJGynnzJdDpK-58_ZXmE9",
  "1ZzqZ3mQcIGkb2Oef52xzc49SsSQ4a1Zz",
  "11YAwK7m01BpQLNod4VHIiW9q3NJKWlII",
  "1DWrtSOQED9wSfioF9Kv0bCQLB-T6-_1w",
  "1YQmOr_Wg0BVUw0HfS08xwVbzmij32Uzb",
  "1gG-Q7kV4APpuwkN-QnkZ_ihQ6nSPUo9f",
  "16kRCREuw9SR1_CR4VHO3FZ03GbaTZbfH",
  "1-fcHHWP5OfEvAzY8KmEd3jEJ6eV0uhgH",
  "1f3ddX1ZWpEeV2Dn6BeYQGL_Ti_8Jo85m",
  "1xOASe0bSVn9umDhrIJdgayqdaHn3R4a5",
  "18BGo9c18B347zdAh0q9gI2Wc1ugOHce-",
  "1wLYS-JAcWi_l_zGF56bmPOKG8JAT7eAG",
  "1LhxrQy-4_zP33UmA98R2gxt_G1nPYixS",
  "1Rd_E9G8XBL03AICQ_Jda_xDGbl2OeX9k",
  "1IaY8iMm_MpvlcPrJ-5z7ufw7zqk3n47h",
  "19SuD5wsqJD8iTWfOEF7KKf7t5TbFQ9qn",
  "1G0Ia2gYwobRIxgL9XoOaClfS9lB6nsWL",
  "1u7DtCRG5wasZZ3a8vII9hFRbeTEClpOY",
  "1KLzN1sXUjEU738ECYQn6qsWQ5kY2GFiM",
  "1ZifdcZ_6f-Xe8yJJ-95iiQNa2tluMVp3",
  "1cxTB-zDKQyjVURKE8rpH0XdMUmILet_b",
  "1YGuIJANVw4tknXNjUa-eGfBi124phwS4",
  "1BoKJcVBU9PmiQ5wQ3wLVQ8-ldWQ3IWmJ",
  "1r2JFUUo60fC2ukD8_4fwXogfU5S40Uke",
  "1n3024dH_H-KTXWZts9oLcYqoBe4eUJhV",
  "1Tb58oc_MTGczpk-f8qYM4pxYuI2yMfYV",
  "1OVA4svhER2bh2DrRM4lqIQc_RWipnp7A",
  "1zlVJC0TQLiU15mfymQvkX5GoMU-GWO2E",
  "1MJlL8Fxd7FsHXMfg6Wtli7_nIKv3Km_1",
  "1x9bBIxQqVrbbFmE4T9YLi_H4et1wb4Pc",
  "10AVMM5bNRXQbTWCWCLGF1NbpjhyCZdwW",
  "1Be0jN81rSme4Dm5gycFgYdMDUhKHrPoj",
  "1Aukb8dR20tBr0KjP9FpH-jCOF6253-qt",
  "1z9Ja2VgX19cGRNiPGFhLeq2FZ3JXNzfK",
  "1k5r_C5_59oPdf5nebY4r83icsnlpBn2x",
  "1lsAVqf1izINQ6NDk6Q1wrPMeQuKO7JAR",
  "1wwsu7zKtqZRZP-_tEFOCg-kS0gfSm2c4",
  "1i1-dUjpgkbPqYdVfZev-NDz-LZVmEAWr",
  "14p2YkjyBRLo5hcNc2JGHf-eKirzszxdw",
  "1QWygGUfZT8vqTYMrMH5Y_uWefx9ePWos",
  "18rKyMxxzXoSvVriAr_1K5zckhXmakJTr",
  "1JVHH_C78_mYVfYKN3I2-ZbC3C3e1iTSg",
  "1LZQ7-PPSFXPOzBe0IoqcIkUOATA7h10b",
  "1P2m_gTp3Lnk42UvO26S49T0tE6oWB0ai",
  "1xCskqi9S2eNyshgRfI_6ro4qQjexfq1C",
  "1Nn6-96Fm19IrT33yLzBz3SVR2kbv759k",
  "1WIX437iscfu7tHBfmd8zft8e51nT0wM8",
  "17Z_nfielAeVyENnWNzaX6uSzml5aUAPk",
  "1FBwuaPnHTPt60UZyHy4tO5UQjlhcfU4Z",
  "1cIDXRX0dJym85rbDc1Du5um7LJ0KKpP5",
  "1o9vxsLwKLODYBmt9jo2sueDy1wUQssVd",
  "1WQxgjc208r5vU0MTLnaq58rWbL9-EP2B",
  "1aPdFQ9VLme0UxuSP0Gl0C1035PmxDXnu",
  "1pGQTTf8rfPPHmMWHX5dgSBRjfxcjuBkx",
  "1nBAsmUd846jjdMPnAFIEmTRnAbblcLv1",
  "1yz5DjPwloytQrSIkLRbiPtNg0Z4WpqB6",
  "1iZc8qR1kl4jIKoio3ZM3nmlCCU6R38YD",
  "12-m826sh8RXxyCyt1eL81lx-DE8BqRv9",
  "1v6Md85IAgzbQFDUxMk_uDGICDAhuWvr2",
  "1FILiElOcwSU_BeYaom4us2xPoknwu276",
  "1lJwn2TexMvS8OJFTmJROt1yblc7DB7ag",
  "100lllMgO0SE-EzA-46iY3riQmlq8C68T",
  "1t16W1a5ClHhVe3duXgMu1ForqnqNBSps",
  "1qJr8j1kt15DfZF-My-Fgh2VcMUDfWlUM",
  "1wYP-F0zVyI4wglQMh66xX8QDePk6igIy",
  "1-0RzJuyJ86Ajz9HYj0aD41N2oNb5MZgE",
  "1EGVFFNuvHqV2MCVA0RQC1ga6z5sBPDQy",
  "1qXIyFcvNXIbxdRybTTAYgWUFqsNhugqH",
  "17ILPx5TCkUSLTipBMQp62JSShCB_4um_",
  "1zGe776gpYInZ1QpAdh3dEEMzWDa4QTav",
  "13iRZrrOXrNie7NiEYENW373g4_GR9zX8",
  "1AnBbD988IHHh4Lc7OT0HtFw5QobNMDxB",
  "1RXQIIbst0hqRJzZ26o2HgivEhEWEJ5wD",
  "1nQ47fEj2jAzU6JBFWN56cpsf7zIQHCq4",
  "16uMfIN2xRucgaMLVZ63fsGwUauwAT_w9",
  "177RXqOswIbda16lPRX0CqxpKortIA1s1",
  "1cjabhagM57srTt9R15V9h65bzG_oNnyB",
  "1miyIaBwtIeUEdFY3Mo6IgQ9Lra9uFNLJ",
  "1lqmqfeT_BWGbGecbJlhqM-7QBLb3eGkR",
  "1y7ACHwW1p0yxwkkutpBZ2CQuUvvJ6TYe",
  "1sSNeHxDfHVoJELVmA2C39FKGmS-qLUgE",
  "12mPGlbBNH50dJ8OpLDNtzQdkaKBM6HRR",
  "1zpGyWX3-himLSjmzQeqKK89NAIrxMRkr",
  "1tfnQucCeRC6LazqPf8r1NrukL1-gMslD",
  "1nWYeLahSJGHXZtyeFJmA2WTi9tJEcbKy",
  "1h60IIca20IJtGqCJRs-JtuR3TSylChwA",
  "168zy4SWrPIZXjaJkvBzqWNxfmGQuwnKE",
  "1mFQFS920xdJgKyDwN65OoVNI8tergGcW",
  "1aCy5f-BckJZjFAokylRoR_3rxPxpYE60",
  "1AEOrpcZHGySi9sVhoGz7OXPeoajOMY65",
  "15PwfQgtOigm5IlR3NeSZqqDFsXJMPXAp",
  "1NS7HVFK4bzJbLMGiEzREzX8lwF0d5BCH",
  "1CIcuWtblcB6ZPxRmuhQVd8LGUBsp7qXL",
  "10G8D8QVVO3WtAOmU057Rjx93jmjlf0nA",
  "1smdjA9awfhyoQpXCxPuWOBu6FRndFcaH",
  "1sO_xJUuuKtsaCAE_CLngCu_74AC5d9um",
  "1_yxje-XtwNK4JELqvLp7EtVPsKXnqN9V",
  "1hOIlPF1JLc6FftMwPPxOt62h0IGHYTyM",
  "1QUWO2dEC-R6SkyqhjxQHnGrE20hj_JY7",
  "1yIthGa4WFgXRNf989W9UQuxKXPhpCN9W",
];

// Master spreadsheet and single tab for all responses
const RESPONSES_SPREADSHEET_ID = "1pKb32Pa4P10iOzL0lz64x6lLM2s7L0znaKU5NTtsJGM"; // REQUIRED: set this to your master spreadsheet ID
const MASTER_SHEET_NAME = "AllResponses";

// Optional: use a prebuilt template Form (with demographics, etc.) and clone it per participant
const TEMPLATE_FORM_ID = "";
// Optional: reduce item creation calls by omitting section headers; put prompt in help text only
const SIMPLE_TASK_ITEMS = true;

function loadBundle(fileId) {
  const id = String(fileId || "").trim();
  const file = DriveApp.getFileById(id);
  const text = file.getBlob().getDataAsString("utf-8");
  return JSON.parse(text);
}

// ---- formatting helpers ----
function stripMarkdown(s) {
  if (!s) return "";
  return s.replace(/^#+\s*/gm, "").trim();
}
function bullets(s) {
  return s.replace(/^[ \t]*-\s/gm, "• ");
}
function splitTopLevelByComma(s) {
  const out = [];
  let depth = 0,
    buf = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === "(") depth++;
    if (ch === ")") depth = Math.max(0, depth - 1);
    if (ch === "," && depth === 0) {
      out.push(buf.trim());
      buf = "";
      continue;
    }
    buf += ch;
  }
  if (buf.trim()) out.push(buf.trim());
  return out;
}
function trimOuterParens(s) {
  s = s.trim();
  if (!s.startsWith("(") || !s.endsWith(")")) return s;
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "(") depth++;
    else if (s[i] === ")") depth--;
    if (depth === 0 && i < s.length - 1) return s;
  }
  return s.slice(1, -1).trim();
}
function formatArgumentBlock(argRaw) {
  const turn = argRaw.includes("⊨") ? "⊨" : argRaw.includes("|=") ? "|=" : null;
  if (!turn) return argRaw.trim();
  const [leftRaw, rightRaw] = argRaw.split(turn);
  const left = leftRaw.trim().replace(/\s+/g, " ");
  const right = rightRaw.trim();
  const items = splitTopLevelByComma(left).map(trimOuterParens);
  const premises = items.join(",\n");
  const conclusion = trimOuterParens(right);
  return `${premises}\n\n⊨ ${conclusion}`;
}
function formatCountermodel(q) {
  if (!q || typeof q.text !== "string") return q;
  const t = stripMarkdown(q.text);
  const tag = "Argument:";
  const idx = t.indexOf(tag);
  if (idx === -1) return { id: q.id, text: bullets(t) };
  const pre = t.slice(0, idx).trim();
  const arg = t.slice(idx + tag.length).trim();
  const pretty = bullets(pre) + "\n\nArgument:\n" + formatArgumentBlock(arg);
  return { id: q.id, text: pretty };
}

// ---- Master sheet helpers ----
function ensureMasterSheet() {
  if (!RESPONSES_SPREADSHEET_ID)
    throw new Error(
      "Set RESPONSES_SPREADSHEET_ID to collect responses in one tab."
    );
  const ss = SpreadsheetApp.openById(RESPONSES_SPREADSHEET_ID);
  let sheet = ss.getSheetByName(MASTER_SHEET_NAME);
  if (!sheet) sheet = ss.insertSheet(MASTER_SHEET_NAME);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(["Timestamp", "FormTitle", "FormId", "ResponseJSON"]);
  }
  return sheet;
}

function ensureFormSubmitTrigger(form) {
  const formId = form.getId();
  const triggers = ScriptApp.getProjectTriggers();
  for (var i = 0; i < triggers.length; i++) {
    var t = triggers[i];
    if (
      t.getHandlerFunction() === "handleFormSubmit" &&
      t.getEventType &&
      t.getEventType() === ScriptApp.EventType.ON_FORM_SUBMIT &&
      t.getTriggerSource &&
      t.getTriggerSource() === ScriptApp.TriggerSource.FORM &&
      t.getTriggerSourceId &&
      t.getTriggerSourceId() === formId
    ) {
      return; // already installed
    }
  }
  ScriptApp.newTrigger("handleFormSubmit")
    .forForm(form)
    .onFormSubmit()
    .create();
}

function handleFormSubmit(e) {
  try {
    const form = e && e.source;
    const response = e && e.response;
    if (!form || !response) return;
    const ts = response.getTimestamp();
    const formTitle = form.getTitle();
    const formId = form.getId();

    const itemResponses = response.getItemResponses();
    const demographics = {};
    const responsesGrouped = {
      symbolization: {},
      countermodel: {},
      validity: {},
    };

    for (let i = 0; i < itemResponses.length; i++) {
      const ir = itemResponses[i];
      const item = ir.getItem();
      const title = item.getTitle();
      const help =
        (typeof item.getHelpText === "function" && item.getHelpText()) || "";
      const value = ir.getResponse();

      if (help.indexOf("Symbolization Task:") === 0) {
        responsesGrouped.symbolization[title] = value;
      } else if (help.indexOf("Countermodel Task:") === 0) {
        responsesGrouped.countermodel[title] = value;
      } else if (help.indexOf("Validity Task:") === 0) {
        responsesGrouped.validity[title] = value;
      } else {
        demographics[title] = value;
      }
    }

    const payload = {
      demographics: demographics,
      responses: responsesGrouped,
    };

    const sheet = ensureMasterSheet();
    sheet.appendRow([ts, formTitle, formId, JSON.stringify(payload)]);
  } catch (err) {
    Logger.log("handleFormSubmit error: " + err);
  }
}

// ---- Form building ----
function addDemographics(form) {
  form.setDescription(
    "This anonymous, voluntary study compares human and AI performance on logic problems.\n" +
      "Do your best using only pencil/paper and your own reasoning.\n" +
      "Do not use web search, AI tools, textbooks/notes, or help from others.\n"
  );
  const pledge = form
    .addMultipleChoiceItem()
    .setTitle("Honor pledge")
    .setHelpText(
      "I will not use web search, AI tools, or other computer aids while answering."
    );
  pledge.setChoices([
    pledge.createChoice("Yes, I agree", true),
    pledge.createChoice("No, I do not agree"),
  ]);
  pledge.setRequired(true);

  form.addTextItem().setTitle("University");
  form.addTextItem().setTitle("Major");
  form
    .addMultipleChoiceItem()
    .setTitle("Year in university")
    .setChoiceValues(["Freshman", "Sophomore", "Junior", "Senior", "Other"]);
  form
    .addMultipleChoiceItem()
    .setTitle("Have you completed an intro logic course?")
    .setChoiceValues(["Yes", "No"])
    .setRequired(true);
  form
    .addMultipleChoiceItem()
    .setTitle("Grade in intro logic (optional)")
    .setChoiceValues(["A", "B", "C", "D", "F", "Prefer not to say"]);
  form
    .addScaleItem()
    .setTitle("Self-rated formal logic comfort")
    .setBounds(1, 5)
    .setLabels("Novice", "Very comfortable");
}
function addTask(form, typeLabel, q, hint) {
  const prompt = bullets(stripMarkdown(q.text));
  if (!SIMPLE_TASK_ITEMS) {
    form.addSectionHeaderItem().setTitle(typeLabel).setHelpText(prompt);
  }
  const help = SIMPLE_TASK_ITEMS ? `${typeLabel}:\n${prompt}` : hint;
  form
    .addParagraphTextItem()
    .setTitle(q.id)
    .setHelpText(help)
    .setRequired(true);
}

// ---- Main ----
function generateFormsForNewParticipants() {
  if (!BUNDLE_FILE_IDS || !BUNDLE_FILE_IDS.length)
    throw new Error("BUNDLE_FILE_IDS is empty – add bundle file IDs.");
  if (!RESPONSES_SPREADSHEET_ID)
    throw new Error(
      "Set RESPONSES_SPREADSHEET_ID to collect all responses in one tab."
    );

  // Ensure master sheet exists
  const masterSheet = ensureMasterSheet();
  const masterUrl = masterSheet.getParent().getUrl();

  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName("Participants");
  const data = sheet.getDataRange().getValues();

  // Pick one bundle per run
  const chosenId =
    BUNDLE_FILE_IDS[Math.floor(Math.random() * BUNDLE_FILE_IDS.length)];
  const bundle = loadBundle(chosenId);

  for (let i = 1; i < data.length; i++) {
    const email = data[i][0];
    const existingFormUrl = data[i][1];
    if (!email || existingFormUrl) continue;

    const sampleCountermodel = bundle.countermodel.map(formatCountermodel);
    const sampleSymbolization = bundle.symbolization;
    const sampleValidity = bundle.validity;

    // Create form (we do NOT set a spreadsheet destination to avoid per-form tabs)
    let form;
    if (TEMPLATE_FORM_ID) {
      const copy = DriveApp.getFileById(TEMPLATE_FORM_ID).makeCopy(
        `Survey for ${email}`
      );
      form = FormApp.openById(copy.getId());
    } else {
      form = FormApp.create(`Survey for ${email}`);
    }

    // Enforce one response per user and hide resubmit/edit
    form.setLimitOneResponsePerUser(true);
    form.setAllowResponseEdits(false).setShowLinkToRespondAgain(false);
    form.setConfirmationMessage(
      "Thank you for supporting our research. Your response has been recorded."
    );

    // Store URLs in Participants: form URL, master sheet URL, timestamp
    const formUrl = form.getPublishedUrl();
    sheet
      .getRange(i + 1, 2, 1, 3)
      .setValues([[formUrl, masterUrl, new Date()]]);

    // Ensure submit trigger to append to master tab
    ensureFormSubmitTrigger(form);

    // Build content
    if (!TEMPLATE_FORM_ID) addDemographics(form);
    sampleSymbolization.forEach((q) =>
      addTask(
        form,
        "Symbolization Task",
        q,
        "Enter a single well-formed formula."
      )
    );
    sampleCountermodel.forEach((q) =>
      addTask(form, "Countermodel Task", q, "Enter a countermodel.")
    );
    sampleValidity.forEach((q) =>
      addTask(
        form,
        "Validity Task",
        q,
        "Enter the numbers of all statements that must be true, separated by commas (e.g., 2,3). If none must be true, write ‘none’."
      )
    );

    // Email link
    MailApp.sendEmail({
      to: email,
      subject: "Logic Study – Your Survey Link",
      body: `Hello,

Thank you for volunteering to take part in our logic study.
Your participation will help us compare human and AI performance on formal logic problems.

Please do your best using only pencil and paper and your own reasoning.
Do not use web search, AI tools, textbooks/notes, or assistance from others.

There is no strict time limit, but please set aside at least 30 minutes
so you can work carefully without interruption.

Click the link below to begin your survey:
${formUrl}

We appreciate your help with this research!

Best,
The JabberBench Research Team`,
    });
  }
}
