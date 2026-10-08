-- (1) drop embedded images (figures are separate files in the OL workflow; captions stay)
-- (2) turn the few inline math snippets of the submitted text into plain Unicode text so that a
--     word-level comparison sees them as text (OMML equations are opaque objects to Word's Compare)
function Image(el) return {} end

local pairs_list = {
  { "\\pm", "±" }, { "\\rightarrow", "→" }, { "\\Delta", "Δ" }, { "\\kappa", "κ" },
  { "\\geq", "≥" }, { "\\times", "×" }, { "\\varepsilon", "ε" }, { "\\in", "∈" },
  { "\\{", "{" }, { "\\}", "}" }, { "\\_", "_" }, { "\\,", " " },
}

local function plain_replace(s, a, b)
  local out, i = {}, 1
  while true do
    local j, k = string.find(s, a, i, true)
    if not j then break end
    out[#out + 1] = string.sub(s, i, j - 1)
    out[#out + 1] = b
    i = k + 1
  end
  out[#out + 1] = string.sub(s, i)
  return table.concat(out)
end

function Math(el)
  local t = el.text
  t = string.gsub(t, "\\mathrm{([^}]*)}", "%1")
  for _, p in ipairs(pairs_list) do t = plain_replace(t, p[1], p[2]) end
  t = string.gsub(t, "%s+", " ")
  return pandoc.Str(t)
end
