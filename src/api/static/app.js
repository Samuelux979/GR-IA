// Logica de la pagina de demo de GR-IA.
// Envia la imagen al endpoint /predict y renderiza los resultados.

const $ = (id) => document.getElementById(id);

const imageInput = $("image-input");
const uploadText = $("upload-text");
const preview    = $("preview");
const topK       = $("top-k");
const generate   = $("generate");
const includeAI  = $("include-ai");
const btn        = $("predict-btn");
const statusBox  = $("status");
const errorBox   = $("error");
const results    = $("results");
const ingDiv     = $("ingredients");
const recipesDiv = $("recipes");
const aiSection  = $("ai-section");
const aiDiv      = $("ai-recipe");

let lastFile     = null;
let lastAIRecipe = null;

// --- Subida de imagen --------------------------------------------------------
imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    lastFile = file;
    uploadText.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    btn.disabled = false;
});

// --- Llamada al endpoint /predict --------------------------------------------
btn.addEventListener("click", async () => {
    if (!lastFile) return;

    setBusy(true);
    hideError();
    results.hidden = true;
    aiSection.hidden = true;
    lastAIRecipe = null;

    const form = new FormData();
    form.append("image", lastFile);

    const params = new URLSearchParams({
        top_k:      topK.value,
        generate:   generate.checked,
        include_ai: includeAI.checked,
    });

    try {
        const resp = await fetch(`/predict?${params}`, {
            method: "POST",
            body:   form,
        });

        if (!resp.ok) {
            const detail = await resp.json().catch(() => ({ detail: "Error desconocido" }));
            throw new Error(detail.detail || `HTTP ${resp.status}`);
        }

        renderResults(await resp.json());
    } catch (err) {
        showError(err.message);
    } finally {
        setBusy(false);
    }
});

// --- Render de resultados ----------------------------------------------------
function renderResults(data) {
    renderIngredients(data.ingredients);
    renderRecipes(data.recipes);

    if (data.ai_recipe) {
        lastAIRecipe = data.ai_recipe;
        renderAIRecipe(data.ai_recipe);
        aiSection.hidden = false;
    }

    if (data.warnings && data.warnings.length) {
        statusBox.textContent = data.warnings.join(" · ");
        statusBox.hidden = false;
    }

    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth" });
}

function renderIngredients(ing) {
    const levels = [
        ["main",          "Principal"],
        ["secondary",     "Secundario"],
        ["accompaniment", "Acompañamiento"],
        ["spices",        "Especias"],
    ];

    ingDiv.innerHTML = "";
    for (const [key, label] of levels) {
        const items = ing[key] || [];
        if (!items.length) continue;
        const row = document.createElement("div");
        row.className = "ingredient-group";
        row.innerHTML = `<span class="label">${label}</span>` +
                        items.map(i => `<span class="chip">${escape(i)}</span>`).join("");
        ingDiv.appendChild(row);
    }
}

function renderRecipes(recipes) {
    recipesDiv.innerHTML = "";
    if (!recipes.length) {
        recipesDiv.innerHTML = "<p>No se encontraron recetas para estos ingredientes.</p>";
        return;
    }

    for (const r of recipes) {
        const isAI       = r.source === "ai_generated";
        const aiBadge    = isAI ? `<span class="badge-ai">IA</span>` : "";
        const macroLabel = r.macros_known === false ? "Valores estimados via USDA" : null;

        const el = document.createElement("div");
        el.className = "recipe";
        el.innerHTML = `
            <div class="recipe-header">
                <div class="recipe-title">${escape(r.title)} ${aiBadge}</div>
                <div class="recipe-meta">
                    ${escape(r.category || "")}
                    <span class="grade grade-${r.match_level}" title="Afinidad con los ingredientes detectados">
                        ${r.grade}/10
                    </span>
                </div>
            </div>
            ${r.matches.length ? `<div class="recipe-matches">Coincide con tu foto: <strong>${r.matches.map(escape).join(", ")}</strong></div>` : ""}
            <div class="recipe-section">
                <h4>Ingredientes</h4>
                <ul>${r.ingredients.map(i => `<li>${escape(i)}</li>`).join("")}</ul>
            </div>
            ${r.steps.length ? `
            <div class="recipe-section">
                <h4>Pasos</h4>
                <ol>${r.steps.map(s => `<li>${escape(s)}</li>`).join("")}</ol>
            </div>` : ""}
            ${formatMacros(r.macros, macroLabel)}
        `;
        recipesDiv.appendChild(el);
    }
}

function formatMacros(m, label) {
    if (m.calories === null || m.calories === undefined) return "";
    const safe = (v) => (v === null || v === undefined ? "?" : v.toFixed(1));
    const note = label ? `<div class="macros-note">${escape(label)}</div>` : "";
    return `
        <div class="macros">
            ${Math.round(m.calories)} kcal ·
            P ${safe(m.protein_g)}g ·
            G ${safe(m.fat_g)}g ·
            C ${safe(m.carbs_g)}g ·
            Fibra ${safe(m.fiber_g)}g
        </div>${note}
    `;
}

function renderAIRecipe(ai) {
    aiDiv.innerHTML = `
        <div class="recipe-title">${escape(ai.title)} <span class="badge-ai">IA</span></div>
        <div class="recipe-section">
            <h4>Ingredientes</h4>
            <ul>${ai.ingredients.map(i => `<li>${escape(i)}</li>`).join("")}</ul>
        </div>
        <div class="recipe-section">
            <h4>Pasos</h4>
            <ol>${ai.steps.map(s => `<li>${escape(s)}</li>`).join("")}</ol>
        </div>
        ${formatMacros(ai.macros, "Valores estimados via USDA")}
        <button id="save-btn">Guardar receta en la base de datos</button>
    `;

    $("save-btn").addEventListener("click", saveAIRecipe);
}

// --- Guardado de receta IA ---------------------------------------------------
async function saveAIRecipe() {
    if (!lastAIRecipe) return;
    const saveBtn = $("save-btn");
    saveBtn.disabled = true;
    saveBtn.textContent = "Guardando...";

    try {
        const resp = await fetch("/save", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(lastAIRecipe),
        });
        if (!resp.ok) {
            const detail = await resp.json().catch(() => ({}));
            const msg = typeof detail.detail === "string"
                ? detail.detail
                : "No se pudo guardar la receta.";
            throw new Error(msg);
        }
        const data = await resp.json();
        saveBtn.textContent = data.duplicate
            ? `Ya existia (id ${data.recipe_id})`
            : `Guardada (id ${data.recipe_id})`;
    } catch (err) {
        saveBtn.disabled = false;
        saveBtn.textContent = "Guardar receta en la base de datos";
        showError(err.message);
    }
}

// --- Utilidades --------------------------------------------------------------
function setBusy(busy) {
    btn.disabled = busy || !lastFile;
    if (busy) {
        statusBox.innerHTML = `<span class="spinner"></span> Procesando imagen...`;
        statusBox.hidden = false;
    } else {
        statusBox.hidden = true;
    }
}

function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
}

function hideError() {
    errorBox.hidden = true;
}

function escape(s) {
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}
