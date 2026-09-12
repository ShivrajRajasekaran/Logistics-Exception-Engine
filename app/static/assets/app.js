/* Logistics Exception Engine - demo console behaviour.
   Kept out of index.html so the markup stays readable and this stays lintable.
   Talks only to this service's own API; no external requests. */

(function () {
  "use strict";

  var CLASS_COLOUR = { "package": "#15803d", "damaged-package": "#dc2626" };
  var FALLBACK_COLOUR = "#2563eb";

  // Curated sample order and captions. Anything else the server reports is
  // appended, so the gallery never silently hides a bundled file.
  var SAMPLES = [
    { file: "intact_parcel.jpg", caption: "intact → CLEAR", name: "Intact parcel" },
    { file: "damaged_parcel.jpg", caption: "clear damage → flagged", name: "Clearly damaged parcel" },
    { file: "ambiguous_parcel.jpg", caption: "borderline → refuses", name: "Borderline parcel" }
  ];
  var DEFAULT_SAMPLE = "damaged_parcel.jpg";

  var PRESETS = [
    ["Needs the detector", "Is this parcel damaged?"],
    ["Counting", "How many parcels are visible?"],
    ["Record only, no inference", "Which carrier handled this in transit?"],
    ["Not trained for it", "How many people are in this image?"],
    ["Off topic", "What is the weather in Chennai?"]
  ];

  var $ = function (id) { return document.getElementById(id); };
  var selected = null;   // File chosen for the single-image panel

  /* ---------- helpers ---------- */

  // FastAPI returns `detail` as a string for HTTPException and as a list of
  // objects for pydantic validation errors. Render both without [object Object].
  function describeError(body, res) {
    var d = body && body.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d.map(function (e) {
        var where = e.loc ? e.loc[e.loc.length - 1] + ": " : "";
        return where + (e.msg || "");
      }).join("; ");
    }
    return "HTTP " + res.status;
  }

  function colourFor(label) { return CLASS_COLOUR[label] || FALLBACK_COLOUR; }

  /* ---------- boxes drawn onto a canvas ---------- */

  function drawDetections(canvas, imageSource, detections, revoke) {
    var img = new Image();
    img.onload = function () {
      var cx = canvas.getContext("2d");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      cx.drawImage(img, 0, 0);

      var lw = Math.max(2, Math.round(canvas.width / 300));
      var fs = Math.max(12, Math.round(canvas.width / 38));
      cx.lineWidth = lw;
      cx.font = "600 " + fs + "px system-ui, sans-serif";
      cx.textBaseline = "top";

      detections.forEach(function (d) {
        var x1 = d.bbox[0], y1 = d.bbox[1], x2 = d.bbox[2], y2 = d.bbox[3];
        var colour = colourFor(d.label);
        cx.strokeStyle = colour;
        cx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        var tag = d.label + " " + d.confidence.toFixed(3);
        var tw = cx.measureText(tag).width;
        var th = fs * 1.45;
        var ty = (y1 - th < 0) ? y1 : y1 - th;   // keep the label on-canvas
        cx.fillStyle = colour;
        cx.fillRect(x1 - lw / 2, ty, tw + fs * 0.7, th);
        cx.fillStyle = "#fff";
        cx.fillText(tag, x1 + fs * 0.28, ty + th * 0.18);
      });
      if (revoke) URL.revokeObjectURL(img.src);
    };
    img.src = imageSource;
  }

  /* ---------- health ---------- */

  fetch("/health").then(function (r) { return r.json(); }).then(function (h) {
    $("dot").className = "dot " + (h.model_loaded ? "up" : "down");
    $("modelState").textContent = h.model_loaded ? "model ready" : "model NOT loaded";
    $("chipClasses").innerHTML = "detects <b>" + (h.classes || []).join(", ") + "</b>";
    $("chipGuard").innerHTML = "guardrail <b>" + h.guardrail_threshold + "</b>";
  }).catch(function () {
    $("dot").className = "dot down";
    $("modelState").textContent = "API unreachable";
  });

  /* ---------- sample tiles ---------- */

  fetch("/api/v1/samples").then(function (r) { return r.json(); }).then(function (data) {
    var tiles = $("tiles");
    var picker = $("imageSelect");
    var known = SAMPLES.filter(function (s) { return data.samples.indexOf(s.file) !== -1; });
    var extra = data.samples.filter(function (n) {
      return !SAMPLES.some(function (s) { return s.file === n; });
    }).map(function (n) { return { file: n, caption: n, name: n }; });

    known.concat(extra).forEach(function (s) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tile";
      btn.setAttribute("aria-pressed", "false");
      btn.innerHTML =
        '<img src="/samples/' + s.file + '" alt="' + s.name + '">' +
        '<span class="cap">' + s.caption + "</span>";
      btn.addEventListener("click", function () {
        Array.prototype.forEach.call(document.querySelectorAll(".tile"), function (t) {
          t.setAttribute("aria-pressed", "false");
        });
        btn.setAttribute("aria-pressed", "true");
        fetch("/samples/" + s.file).then(function (r) { return r.blob(); }).then(function (blob) {
          selected = new File([blob], s.file, { type: blob.type });
          $("runBtn").disabled = false;
          $("dropText").innerHTML = "<strong>" + s.file + "</strong>" +
            '<span class="hint">bundled sample · click Run detection</span>';
        });
        picker.value = "sample_images/" + s.file;
      });
      tiles.appendChild(btn);

      var opt = document.createElement("option");
      opt.value = "sample_images/" + s.file;
      opt.textContent = s.name + "  (" + s.file + ")";
      picker.appendChild(opt);
    });

    var none = document.createElement("option");
    none.value = "";
    none.textContent = "(no image — answer from the record alone)";
    picker.appendChild(none);
    if (data.samples.indexOf(DEFAULT_SAMPLE) !== -1) {
      picker.value = "sample_images/" + DEFAULT_SAMPLE;
    }
  });

  /* ---------- preset questions ---------- */

  PRESETS.forEach(function (pair) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.innerHTML = '<span class="tag">' + pair[0] + "</span>" + pair[1];
    btn.addEventListener("click", function () {
      $("question").value = pair[1];
      $("askBtn").click();
    });
    $("presets").appendChild(btn);
  });

  /* ---------- file intake: one image, or many ---------- */

  var dropzone = $("dropzone");
  var fileInput = $("fileInput");

  dropzone.addEventListener("click", function () { fileInput.click(); });
  dropzone.addEventListener("dragover", function (e) {
    e.preventDefault();
    dropzone.classList.add("over");
  });
  dropzone.addEventListener("dragleave", function () { dropzone.classList.remove("over"); });
  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropzone.classList.remove("over");
    accept(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", function (e) { accept(e.target.files); });

  function accept(fileList) {
    var files = Array.prototype.slice.call(fileList).filter(function (f) {
      return f.type.indexOf("image/") === 0;
    });
    if (!files.length) return;

    Array.prototype.forEach.call(document.querySelectorAll(".tile"), function (t) {
      t.setAttribute("aria-pressed", "false");
    });

    if (files.length === 1) {
      selected = files[0];
      $("runBtn").disabled = false;
      $("dropText").innerHTML = "<strong>" + files[0].name + "</strong>" +
        '<span class="hint">' + (files[0].size / 1024).toFixed(0) +
        " KB · click Run detection</span>";
      return;
    }
    // More than one file is a batch: run the whole folder straight away.
    runBatch(files);
  }

  /* ---------- single image ---------- */

  $("runBtn").addEventListener("click", function () {
    if (!selected) return;
    var btn = $("runBtn");
    $("detectError").textContent = "";
    btn.disabled = true;
    btn.textContent = "Detecting…";

    var form = new FormData();
    form.append("file", selected);

    fetch("/api/v1/detect", { method: "POST", body: form })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(describeError(body, res));
          return body;
        });
      })
      .then(function (data) {
        $("batchPanel").classList.add("hidden");
        $("singlePanel").classList.remove("hidden");
        $("downloadBtn").classList.remove("hidden");
        $("jsonBtn").classList.remove("hidden");

        drawDetections($("canvas"), URL.createObjectURL(selected), data.detections, true);
        $("detectMeta").innerHTML = "<b>" + data.count + "</b> detection(s) · " +
          data.image_size[0] + "×" + data.image_size[1] + " px · " +
          data.inference_time_ms.toFixed(1) + " ms";

        $("detectTable").innerHTML = data.detections.length
          ? "<tr><th>Class</th><th class='num'>Confidence</th><th>Box [x1, y1, x2, y2]</th></tr>" +
            data.detections.map(function (d) {
              return "<tr><td><span class='swatch' style='background:" + colourFor(d.label) +
                "'></span>" + d.label + "</td><td class='num'>" + d.confidence.toFixed(4) +
                "</td><td style='color:var(--muted)'>" +
                d.bbox.map(function (v) { return v.toFixed(1); }).join(", ") + "</td></tr>";
            }).join("")
          : "<tr><td style='color:var(--muted)'>Nothing detected above the 0.25 serving threshold.</td></tr>";
        $("detectJson").textContent = JSON.stringify(data, null, 2);
        loadLog();          // the observation was just recorded; show it
      })
      .catch(function (err) { $("detectError").textContent = "Detection failed: " + err.message; })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "Run detection";
      });
  });

  $("downloadBtn").addEventListener("click", function () {
    var a = document.createElement("a");
    a.download = "detection_" + Date.now() + ".png";
    a.href = $("canvas").toDataURL("image/png");
    a.click();
  });
  $("jsonBtn").addEventListener("click", function () { $("detectJson").classList.toggle("hidden"); });
  $("reasonJsonBtn").addEventListener("click", function () { $("reasonJson").classList.toggle("hidden"); });

  /* ---------- batch: a whole folder of images ---------- */

  function runBatch(files) {
    $("singlePanel").classList.add("hidden");
    $("batchPanel").classList.remove("hidden");
    $("detectError").textContent = "";
    $("batchGrid").innerHTML = "";
    $("dropText").innerHTML = "<strong>" + files.length + " images queued</strong>" +
      '<span class="hint">processing one at a time</span>';

    var tally = { images: 0, damaged: 0, clear: 0, empty: 0, boxes: 0, ms: 0 };
    var bar = $("progressBar");

    // Sequential on purpose: the container holds one model and a burst of
    // parallel requests only queues inside it while inflating peak memory.
    function step(i) {
      if (i >= files.length) {
        $("dropText").innerHTML = "<strong>Done — " + files.length + " images</strong>" +
          '<span class="hint">drop another folder to run again</span>';
        loadLog();          // once at the end, not once per image
        return;
      }
      bar.style.width = Math.round((i / files.length) * 100) + "%";

      var form = new FormData();
      form.append("file", files[i]);
      fetch("/api/v1/detect", { method: "POST", body: form })
        .then(function (res) {
          return res.json().then(function (body) {
            if (!res.ok) throw new Error(describeError(body, res));
            return body;
          });
        })
        .then(function (data) {
          tally.images += 1;
          tally.boxes += data.count;
          tally.ms += data.inference_time_ms;
          var hasDamage = data.detections.some(function (d) { return d.label === "damaged-package"; });
          if (!data.count) tally.empty += 1;
          else if (hasDamage) tally.damaged += 1;
          else tally.clear += 1;

          addResultCard(files[i], data, hasDamage);
          renderTally(tally);
        })
        .catch(function (err) {
          tally.images += 1;
          addErrorCard(files[i], err.message);
          renderTally(tally);
        })
        .finally(function () {
          bar.style.width = Math.round(((i + 1) / files.length) * 100) + "%";
          step(i + 1);
        });
    }
    renderTally(tally);
    step(0);
  }

  function renderTally(t) {
    $("batchSummary").innerHTML = [
      ["images", t.images],
      ["damaged", t.damaged],
      ["clear", t.clear],
      ["avg ms", t.images ? Math.round(t.ms / t.images) : 0]
    ].map(function (pair) {
      return "<div class='stat'><div class='n'>" + pair[1] +
        "</div><div class='k'>" + pair[0] + "</div></div>";
    }).join("");
  }

  function addResultCard(file, data, hasDamage) {
    var card = document.createElement("div");
    card.className = "result";
    var cv = document.createElement("canvas");
    card.appendChild(cv);

    var meta = document.createElement("div");
    meta.className = "rmeta";
    var cls = data.count === 0 ? "none" : (hasDamage ? "dmg" : "ok");
    var top = data.detections.length
      ? data.detections.reduce(function (a, b) { return a.confidence > b.confidence ? a : b; })
      : null;
    meta.innerHTML =
      "<div class='fname' title='" + file.name + "'>" + file.name + "</div>" +
      "<div class='rline " + cls + "'>" +
      (top ? top.label + " " + top.confidence.toFixed(2) : "no detection") + "</div>";
    card.appendChild(meta);
    $("batchGrid").appendChild(card);

    drawDetections(cv, URL.createObjectURL(file), data.detections, true);
  }

  function addErrorCard(file, message) {
    var card = document.createElement("div");
    card.className = "result";
    card.innerHTML = "<div class='rmeta'><div class='fname' title='" + file.name + "'>" +
      file.name + "</div><div class='rline dmg'>" + message + "</div></div>";
    $("batchGrid").appendChild(card);
  }

  /* ---------- reasoning ---------- */

  $("askBtn").addEventListener("click", function () {
    var btn = $("askBtn");
    $("reasonError").textContent = "";
    btn.disabled = true;
    btn.textContent = "Reasoning…";

    var payload = { package_id: $("packageSelect").value, query: $("question").value };
    if ($("imageSelect").value) payload.image_path = $("imageSelect").value;

    fetch("/api/v1/reason", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(describeError(body, res));
          return body;
        });
      })
      .then(function (d) {
        $("verdict").classList.remove("hidden");
        $("reasonJsonBtn").classList.remove("hidden");

        $("statusBadge").textContent = d.status;
        $("statusBadge").className = "badge b-" + d.status;

        var flags = "detector used: <b>" + d.requires_vision_model +
          "</b> · guardrail passed: <b>" + d.guardrail_passed + "</b>";
        if (d.max_critical_confidence !== null && d.max_critical_confidence !== undefined) {
          flags += " · peak damage confidence: <b>" +
            d.max_critical_confidence.toFixed(4) + "</b>";
        }
        $("reasonFlags").innerHTML = flags;
        $("reasonSummary").textContent = d.decision_summary;

        $("fallbackNote").textContent =
          d.decision_summary.indexOf("[deterministic fallback") === 0
            ? "The wording above came from a fixed template, not a language model. The status, " +
              "the guardrail and the detections are computed the same way either way, so the " +
              "decision does not depend on the language model being reachable."
            : "";

        $("reasonTable").innerHTML = (d.detections && d.detections.length)
          ? "<tr><th>Class</th><th class='num'>Confidence</th></tr>" +
            d.detections.map(function (x) {
              return "<tr><td><span class='swatch' style='background:" + colourFor(x.label) +
                "'></span>" + x.label + "</td><td class='num'>" +
                x.confidence.toFixed(4) + "</td></tr>";
            }).join("")
          : "";
        $("reasonJson").textContent = JSON.stringify(d, null, 2);
        loadLog();          // the verdict was just appended; show it
      })
      .catch(function (err) { $("reasonError").textContent = "Reasoning failed: " + err.message; })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "Ask";
      });
  });

  /* ---------- adjudication log ---------- */

  function escapeText(value) {
    var d = document.createElement("div");
    d.textContent = value == null ? "" : String(value);
    return d.innerHTML;
  }

  // Entries come from the append-only log, which records whatever question was
  // asked. That is user-supplied text, so it is escaped rather than trusted.
  function loadStats() {
    return fetch("/api/v1/stats").then(function (r) { return r.json(); })
      .then(function (st) {
        $("storeStats").innerHTML = [
          ["images seen", st.detections],
          ["with damage", st.images_with_damage],
          ["exceptions", st.exceptions_flagged],
          ["refusals", st.refusals]
        ].map(function (p) {
          return "<div class='stat'><div class='n'>" + p[1] +
                 "</div><div class='k'>" + p[0] + "</div></div>";
        }).join("");
      }).catch(function () { $("storeStats").innerHTML = ""; });
  }

  function loadDetections() {
    return fetch("/api/v1/detections?limit=25")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.entries.length) {
          $("detTable").innerHTML = "";
          $("detEmpty").textContent =
            "No detections recorded yet. Run one in Part A and it will appear here.";
          return;
        }
        $("detEmpty").textContent = "";
        $("detTable").innerHTML =
          "<tr><th>When (UTC)</th><th>Image</th><th class='num'>Objects</th>" +
          "<th class='num'>Top conf.</th><th class='num'>ms</th><th>Verdict</th></tr>" +
          d.entries.map(function (e) {
            var conf = (e.max_confidence === null || e.max_confidence === undefined)
              ? "—" : e.max_confidence.toFixed(3);
            var verdict = e.count === 0
              ? "<span class='badge sm b-UNSUPPORTED_CAPABILITY'>nothing found</span>"
              : (e.has_damage
                  ? "<span class='badge sm b-EXCEPTION_FLAGGED'>damage</span>"
                  : "<span class='badge sm b-CLEAR'>intact</span>");
            return "<tr>" +
              "<td class='when'>" + escapeText(e.recorded_at.replace("T", " ").slice(0, 19)) + "</td>" +
              "<td class='query' title='" + escapeText(e.filename) + "'>" +
                escapeText(e.filename || "(unnamed)") + "</td>" +
              "<td class='num'>" + e.count + "</td>" +
              "<td class='num'>" + conf + "</td>" +
              "<td class='num'>" + Math.round(e.inference_time_ms) + "</td>" +
              "<td>" + verdict + "</td></tr>";
          }).join("");
      })
      .catch(function () { $("detEmpty").textContent = "Could not read detections."; });
  }

  function loadLog() {
    loadStats();
    loadDetections();
    return fetch("/api/v1/exceptions?limit=25")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var chip = $("ledgerChip");
        chip.innerHTML = "ledger <b>" + (d.ledger_intact ? "intact" : "MODIFIED") +
          "</b> · " + d.ledger_digest + " · <b>" + d.count + "</b> recorded";
        chip.className = "chip" + (d.ledger_intact ? "" : " bad");

        if (!d.entries.length) {
          $("logTable").innerHTML = "";
          $("logEmpty").textContent =
            "Nothing recorded yet. Ask a question in Part B and it will appear here.";
          return;
        }
        $("logEmpty").textContent = "";
        $("logTable").innerHTML =
          "<tr><th>When (UTC)</th><th>Package</th><th>Status</th>" +
          "<th class='num'>Peak conf.</th><th>Question</th></tr>" +
          d.entries.map(function (e) {
            var conf = (e.max_critical_confidence === null ||
                        e.max_critical_confidence === undefined)
              ? "—" : e.max_critical_confidence.toFixed(3);
            return "<tr>" +
              "<td class='when'>" + escapeText(e.recorded_at.replace("T", " ").slice(0, 19)) + "</td>" +
              "<td>" + escapeText(e.package_id) + "</td>" +
              "<td><span class='badge sm b-" + escapeText(e.status) + "'>" +
                escapeText(e.status) + "</span></td>" +
              "<td class='num'>" + conf + "</td>" +
              "<td class='query' title='" + escapeText(e.query) + "'>" +
                escapeText(e.query) + "</td>" +
            "</tr>";
          }).join("");
      })
      .catch(function () {
        $("logEmpty").textContent = "Could not read the adjudication log.";
      });
  }

  $("refreshLog").addEventListener("click", loadLog);
  loadLog();
})();
