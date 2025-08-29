/* ======================================================================
 * visual.js — BẢN SẠCH HỢP NHẤT (KIS / Q&A / TRAKE nhiều dòng)
 *  - Giữ nguyên UI hiện tại (home.html)
 *  - Không phụ thuộc các bản patch cũ (đã gỡ lặp code)
 *  - Có comment theo từng khối
 * ====================================================================== */

/* =========================================================================
 * (A) TIỆN ÍCH CHUNG
 * ========================================================================= */

/** Form tìm kiếm để build URL Find Similar giữ tham số hiện tại */
const searchForm = document.querySelector('form[action*="textsearch"]');

/** Lấy giá trị input theo name trong form (dùng build URL) */
const getVal = (name) =>
  (searchForm?.querySelector(`input[name="${name}"]`)?.value ?? '').trim();

/** Rút mã video từ path keyframe
 * "data/keyframes/Keyframes_L21/L21_V001/020.jpg" -> "L21_V001"
 */
function extractVcode(path) {
  const parts = String(path).split('/');
  return parts.length >= 2 ? parts[parts.length - 2] : 'unknown';
}

/** Rút chỉ số keyframe n từ tên file
 * ".../007.jpg" -> 7 ; ".../00020.webp" -> 20
 */
function extractN(path) {
  const fname = (String(path).split('/').pop() || '').toLowerCase();
  const m = fname.match(/(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

/** Biến global lưu lựa chọn hiện hành (phục vụ nút Add) */
window.Sel = window.Sel || { vcode: null, frame_idx: null };

/** Đồng hồ Added dùng chung (tuỳ theo mode) */
function setAddedCounter(n) {
  const el1 = document.getElementById('addedCount');
  const el2 = document.getElementById('addedCountPlayer');
  if (el1) el1.textContent = String(n);
  if (el2) el2.textContent = String(n);
}

/* =========================================================================
 * (B) PREVIEW & NÚT SHOW VIDEO
 * ========================================================================= */

/** Khi click 1 card keyframe -> cập nhật preview + ghi lựa chọn cho Add */
window.onPick = function (card) {
  // --- lấy data từ item ---
  const id    = card.getAttribute('data-id');     // id nội bộ (chỉ hiển thị)
  const path  = card.getAttribute('data-path');   // đường dẫn ảnh
  const score = card.getAttribute('data-score');  // điểm xếp hạng

  // --- phần tử panel ---
  const ph    = document.getElementById('pvPh');
  const img   = document.getElementById('pvImg');
  const vid   = document.getElementById('pvVid');
  const idEl  = document.getElementById('pvId');
  const scEl  = document.getElementById('pvScore');
  const tgBtn = document.getElementById('btnToggle');
  const sim   = document.getElementById('btnSimilar');

  if (!img || !vid || !idEl || !scEl || !tgBtn || !sim) return;

  // --- cập nhật ảnh preview ---
  img.src = '/get_img?fpath=' + encodeURIComponent(path);
  img.classList.remove('hidden');
  vid.classList.add('hidden');
  if (ph) ph.classList.add('hidden');
  tgBtn.disabled = false;

  // --- cập nhật text info ---
  idEl.textContent = id ?? '—';
  scEl.textContent = Number(score || 0).toFixed(4);

  // --- build URL Find similar (giữ tham số UI) ---
  const params = new URLSearchParams({
    imgid: id,
    index: 0,
    textquery: getVal('textquery'),
    search_type: getVal('search_type') || 'visual',
    topk: getVal('topk') || '100'
  });
  sim.href = '/imgsearch?' + params.toString();
  sim.removeAttribute('aria-disabled');

  // --- chuẩn bị Show Video & ghi lựa chọn Add ---
  const vcode = extractVcode(path);
  const n     = extractN(path);
  tgBtn.dataset.vcode = vcode;
  tgBtn.dataset.frame = String(n);

  // Ghi lựa chọn toàn cục cho 3 mode
  window.Sel.vcode = vcode;
  window.Sel.frame_idx = n;
  window.currentVcode = vcode;       // dự phòng cho code cũ (nếu có)
  window.currentFrameIdx = n;

  // Bật nút Add (đổi style cho rõ)
  const btnAdd = document.getElementById('btnAddFromPreview');
  if (btnAdd) {
    btnAdd.disabled = false;
    btnAdd.classList.remove('bg-gray-400');
    btnAdd.classList.add('bg-emerald-600','hover:bg-emerald-700','text-white');
  }
};

/** Nút Show video -> mở tab mới tới /watch?vcode=<vcode>&frame=<n> */
document.getElementById('btnToggle')?.addEventListener('click', function () {
  const vcode = this.dataset.vcode;
  const n     = this.dataset.frame || '';
  if (!vcode) return;

  const url = new URL('/watch', window.location.origin);
  url.searchParams.set('vcode', vcode);
  if (n !== '') url.searchParams.set('frame', n);

  window.open(
    url.toString(),
    '_blank',
    'noopener,noreferrer,width=1280,height=720,resizable,scrollbars'
  );
});

/* =========================================================================
 * (C) MODE CHỌN HÌNH THỨC NỘP: KIS | Q&A | TRAKE
 *  - Chặn điều hướng 3 tab, chỉ đổi 'mode' + highlight
 *  - Gắn hành vi cho Add/Submit theo mode
 * ========================================================================= */

const __MODE_KEY = 'submit_mode';
function getMode() {
  const m = (localStorage.getItem(__MODE_KEY) || '').toLowerCase();
  return ['kis','qa','trake'].includes(m) ? m : 'kis';
}
function setMode(m) {
  const mode = (m||'').toLowerCase();
  if (!['kis','qa','trake'].includes(mode)) return;
  localStorage.setItem(__MODE_KEY, mode);
  highlightModeTab(mode);
  // Gợi ý nhỏ cạnh Submit
  const hint = document.getElementById('csvFilenameHint');
  if (hint) hint.textContent = `(${mode.toUpperCase()} mode)`;
  // Rebind hành vi nút theo mode
  rebindActionsForMode();
}
function highlightModeTab(mode) {
  const tabs = {
    kis:   document.querySelector('a[href="/kis"]'),
    qa:    document.querySelector('a[href="/qa"]'),
    trake: document.querySelector('a[href="/trake"]'),
  };
  Object.values(tabs).forEach(a => a && a.classList.remove('border','bg-blue-50','text-blue-700'));
  if (tabs[mode]) tabs[mode].classList.add('border','bg-blue-50','text-blue-700');
}
function wireModeTabs() {
  const bind = (sel, mode) => {
    const el = document.querySelector(sel);
    if (!el) return;
    el.addEventListener('click', (e) => {
      e.preventDefault();
      setMode(mode);
    });
    el.classList.add('cursor-pointer');
  };
  bind('a[href="/kis"]',   'kis');
  bind('a[href="/qa"]',    'qa');
  bind('a[href="/trake"]', 'trake');
}

/* =========================================================================
 * (D) KIS — danh sách dòng, submit qua /export/kis
 * ========================================================================= */

const KIS_ROWS_KEY   = 'kis_rows';
const KIS_FNAME_KEY  = 'kis_fname';
const KIS_MAX_LINES  = 100;

function kis_rows_get() { try { return JSON.parse(localStorage.getItem(KIS_ROWS_KEY)||'[]'); } catch { return []; } }
function kis_rows_set(a) { localStorage.setItem(KIS_ROWS_KEY, JSON.stringify((a||[]).slice(0,KIS_MAX_LINES))); syncAddedCounter(); }
function kis_fname_get() { return localStorage.getItem(KIS_FNAME_KEY) || 'submit_kis.csv'; }
function kis_fname_set(n){ localStorage.setItem(KIS_FNAME_KEY, String(n||'').trim()); }

function kis_addLine(vcode, frameIdx) {
  const rows = kis_rows_get();
  if (rows.length >= KIS_MAX_LINES) return false;
  rows.push({ vcode:String(vcode).trim(), frame_idx: parseInt(frameIdx,10)||0 });
  kis_rows_set(rows);
  return true;
}

/** Thu thập keyframes đang hiển thị (dùng bổ sung cho đủ 100 nếu cần) */
function collectGridFrames() {
  const cards = document.querySelectorAll('.card[data-path][data-frame]');
  const out = [];
  cards.forEach(card => {
    const path  = card.getAttribute('data-path')  || '';
    const frame = card.getAttribute('data-frame');
    const parts = path.split('/').filter(Boolean);
    const vcode = parts.length >= 2 ? parts[parts.length - 2] : null;
    const n = frame != null ? parseInt(frame, 10) : NaN;
    if (vcode && Number.isInteger(n) && n >= 0) {
      out.push({ vcode, frame_idx: n });
    }
  });
  return out;
}

/** Xây 100 dòng: ưu tiên các dòng user đã Add, sau đó bổ sung từ grid */
function kis_build_100_rows() {
  const MAX = 100;
  const added = kis_rows_get();
  const grid  = collectGridFrames();
  const seen = new Set();
  const key = (r) => `${r.vcode}#${r.frame_idx}`;
  const finalRows = [];

  for (const r of added) {
    const k = key(r); if (!seen.has(k)) { finalRows.push(r); seen.add(k); if (finalRows.length>=MAX) return finalRows; }
  }
  for (const r of grid) {
    const k = key(r); if (!seen.has(k)) { finalRows.push(r); seen.add(k); if (finalRows.length>=MAX) return finalRows; }
  }
  if (finalRows.length > 0) while (finalRows.length < MAX) finalRows.push(finalRows[finalRows.length-1]);
  return finalRows;
}

/** Submit KIS: POST lên /export/kis -> tải file CSV */
async function kis_submit() {
  const rows100 = kis_build_100_rows();
  const filename = kis_fname_get();

  const res = await fetch('/export/kis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows: rows100, filename })
  });
  if (!res.ok) {
    const t = await res.text();
    alert('Export failed: ' + t);
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { href:url, download: filename });
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);

  // Dọn nháp KIS sau khi nộp (tuỳ bạn, có thể giữ lại)
  localStorage.setItem(KIS_ROWS_KEY, '[]');
  syncAddedCounter();
}

/* =========================================================================
 * (E) Q&A — nhiều dòng: [{video, frame, answer}] -> CSV client
 * ========================================================================= */

const QA_ROWS_KEY  = 'qa_rows';
const QA_MAX_LINES = 100;

function qa_rows_get(){ try { return JSON.parse(localStorage.getItem(QA_ROWS_KEY)||'[]'); } catch { return []; } }
function qa_rows_set(a){ localStorage.setItem(QA_ROWS_KEY, JSON.stringify((a||[]).slice(0, QA_MAX_LINES))); syncAddedCounter(); }

function qa_addLine(vcode, frameIdx, answer) {
  const rows = qa_rows_get();
  if (rows.length >= QA_MAX_LINES) return false;
  const ans = String(answer ?? '').trim().slice(0, 100); // BTC: ≤100
  rows.push({ video:String(vcode).trim(), frame: parseInt(frameIdx,10)||0, answer: ans });
  qa_rows_set(rows);
  return true;
}

function qa_submit() {
  const rows = qa_rows_get();
  if (!rows.length) return alert('Danh sách trống. Hãy Add ít nhất 1 dòng.');
  const lines = rows.map(it => `${it.video}, ${it.frame}, "${it.answer}"`).join('\n');
  const blob = new Blob(["\uFEFF" + rows.join('\n')], { type:'text/csv;charset=utf-8;' });
  const a = Object.assign(document.createElement('a'), { href:URL.createObjectURL(blob), download:'qa_submission.csv' });
  a.click();
}

/* =========================================================================
 * (F) TRAKE — draft 1 dòng + danh sách nhiều dòng (rows)
 *  - draft: { video, frames:Array(N), N }
 *  - rows:  [{ video, frames:Array(N), N }, ...]
 * ========================================================================= */

const TK_DRAFT_KEY = 'trakeDraft';
const TK_ROWS_KEY  = 'trakeRows';
const TK_MAX_LINES = 100;

/** Draft hiện tại */
function tk_draft_get(){ try { return JSON.parse(localStorage.getItem(TK_DRAFT_KEY) || 'null'); } catch { return null; } }
function tk_draft_set(s){
  localStorage.setItem(TK_DRAFT_KEY, JSON.stringify(s));
  // Đồng hồ Trake hiển thị số DÒNG, không phải số frame -> syncAddedCounter()
  syncAddedCounter();
}

/** Khởi tạo N hành động */
function tk_init(N){
  const n = Math.max(1, parseInt(N,10) || 1);
  tk_draft_set({ video:'', frames:Array(n).fill(null), N:n });
}
/** Gán video hiện hành */
function tk_setVideo(v){
  const s = tk_draft_get() || { video:'', frames:[], N:0 };
  s.video = String(v||'').trim();
  tk_draft_set(s);
}
/** Đặt frame cho hành động i (1..N) */
function tk_setAt(i1, frameIdx){
  const s = tk_draft_get();
  if (!s) return alert('Chưa đặt số hành động (N).');
  const i = parseInt(i1,10);
  if (!(i>=1 && i<=s.N)) return alert(`Chỉ số i phải trong [1..${s.N}].`);
  s.frames[i-1] = parseInt(frameIdx,10)||0;
  tk_draft_set(s);
}

/** Danh sách nhiều dòng */
function tk_rows_get(){ try { return JSON.parse(localStorage.getItem(TK_ROWS_KEY) || '[]'); } catch { return []; } }
function tk_rows_set(rows){
  localStorage.setItem(TK_ROWS_KEY, JSON.stringify((rows||[]).slice(0, TK_MAX_LINES)));
  syncAddedCounter();
}

/** Thêm 1 dòng từ draft vào rows (yêu cầu đủ N frame) */
function tk_addLineFromDraft() {
  const s = tk_draft_get();
  if (!s) return alert('Chưa khởi tạo N. Bấm Set N trước.');
  if (!s.video) return alert('Chưa đặt tên video. Hãy Add ít nhất 1 frame để hệ thống ghi video.');
  if (s.frames.some(x => x == null)) return alert('Chưa đủ N frame cho tất cả hành động.');

  const rows = tk_rows_get();
  if (rows.length >= TK_MAX_LINES) return alert('Đã đủ 100 dòng.');
  rows.push({ video:s.video, frames:s.frames.slice(), N:s.N });
  tk_rows_set(rows);

  // Reset frames để làm dòng kế tiếp (giữ nguyên video/N cho tiện)
  tk_draft_set({ video:s.video, N:s.N, frames:Array(s.N).fill(null) });
}

/** Xuất CSV nhiều dòng: <video>, f1, f2, ..., fN */
function tk_submit_all() {
  const rows = tk_rows_get();
  // Nếu draft đang đủ N mà chưa add -> hỏi chốt luôn
  const s = tk_draft_get();
  if (s && s.video && Array.isArray(s.frames) && !s.frames.some(x => x==null)) {
    const addNow = confirm('Draft hiện tại đã đủ frame. Thêm vào danh sách trước khi Submit không?');
    if (addNow) { rows.push({ video:s.video, frames:s.frames.slice(), N:s.N }); }
  }
  if (!rows.length) return alert('Chưa có dòng nào. Hãy bấm "Add Line" sau khi gán đủ N frame.');
  const lines = rows.map(r => [r.video, ...r.frames].join(', ')).join('\n');
  const blob = new Blob([lines], { type:'text/csv;charset=utf-8;' });
  const a = Object.assign(document.createElement('a'), { href:URL.createObjectURL(blob), download:'trake_submission.csv' });
  a.click();
}

/** Nút Set N (tạo đúng 1 lần, chỉ khi ở Trake) */
function ensureTrakeSetNButton() {
  if (getMode() !== 'trake') return;
  if (document.getElementById('trakeSetN')) return;
  const submitBtn = document.getElementById('btnSubmitCSV'); if (!submitBtn) return;

  const btn = document.createElement('button');
  btn.id = 'trakeSetN';
  btn.textContent = 'Set N';
  btn.className = 'px-3 py-2 rounded-md bg-slate-200 hover:bg-slate-300 text-slate-800 text-sm';
  submitBtn.parentNode.insertBefore(btn, submitBtn);

  btn.addEventListener('click', () => {
    let s = tk_draft_get();
    if (!s) { tk_init(4); s = tk_draft_get(); }
    const curN = s.N || 4;
    const input = prompt('Nhập số hành động (N ≥ 1):', String(curN));
    if (input === null) return;
    const nNew = Math.max(1, parseInt(input, 10) || 1);
    const frames = Array.isArray(s.frames) ? s.frames.slice(0, nNew) : [];
    while (frames.length < nNew) frames.push(null);
    tk_draft_set({ video: s.video || '', frames, N: nNew });

    const hint = document.getElementById('csvFilenameHint');
    if (hint) hint.textContent = `(TRAKE mode, N=${nNew})`;
  });

  // Hiển thị N hiện tại ở hint
  const s0 = tk_draft_get();
  const hint = document.getElementById('csvFilenameHint');
  if (hint) hint.textContent = s0 && s0.N ? `(TRAKE mode, N=${s0.N})` : `(TRAKE mode)`;
}

/** Nút Add Line (tạo đúng 1 lần, chỉ khi ở Trake) */
function ensureTrakeAddLineButton() {
  if (getMode() !== 'trake') return;
  if (document.getElementById('trakeAddLine')) return;
  const submitBtn = document.getElementById('btnSubmitCSV'); if (!submitBtn) return;

  const btn = document.createElement('button');
  btn.id = 'trakeAddLine';
  btn.textContent = 'Add Line';
  btn.className = 'px-3 py-2 rounded-md bg-slate-200 hover:bg-slate-300 text-slate-800 text-sm';
  submitBtn.parentNode.insertBefore(btn, submitBtn);

  btn.addEventListener('click', tk_addLineFromDraft);
}

/* =========================================================================
 * (G) REBIND NÚT ADD / SUBMIT THEO MODE
 *  - Clone & replace để loại bỏ mọi listener cũ.
 * ========================================================================= */

function cloneReplace(el){ if (!el) return null; const c = el.cloneNode(true); el.parentNode.replaceChild(c, el); return c; }

function rebindActionsForMode() {
  const mode = getMode();

  // 1) Nút Add — mỗi mode một hành vi
  let btnAdd = document.getElementById('btnAddFromPreview');
  btnAdd = cloneReplace(btnAdd);
  if (btnAdd) {
    btnAdd.disabled = false;
    btnAdd.addEventListener('click', () => {
      const v = window.Sel?.vcode || window.currentVcode;
      const f = window.Sel?.frame_idx ?? window.currentFrameIdx;
      if (!v || typeof f === 'undefined') return alert('Hãy chọn 1 frame (hoặc pause video) trước khi Add.');

      if (mode === 'kis') {
        if (!kis_addLine(v, f)) alert('Đã đủ 100 dòng.');
      }
      else if (mode === 'qa') {
        const a = prompt('Nhập Answer (VN/EN, ≤100 ký tự):','');
        if (a === null) return;
        if (!qa_addLine(v, f, a)) alert('Đã đủ 100 dòng.');
      }
      else if (mode === 'trake') {
        if (!tk_draft_get()) tk_init(4);            // nếu chưa có N -> mặc định 4
        tk_setVideo(v);                              // luôn set video theo lựa chọn
        const s = tk_draft_get();
        const i = prompt(`Gán vào hành động thứ mấy? (1..${s.N})`, '1');
        if (i) tk_setAt(i, f);
      }
    });
  }

  // 2) Nút Submit — mỗi mode một hành vi
  let btnSubmit = document.getElementById('btnSubmitCSV');
  btnSubmit = cloneReplace(btnSubmit);
  if (btnSubmit) {
    btnSubmit.addEventListener('click', () => {
      if (mode === 'kis')   return kis_submit();
      if (mode === 'qa')    return qa_submit();
      if (mode === 'trake') return tk_submit_all();
    });
  }

  // 3) Nút Add File (đặt tên CSV KIS) — giữ nguyên cho KIS, hiển thị hint cho mode khác
  const btnSetFn = document.getElementById('btnSetFilename');
  const fnHint   = document.getElementById('csvFilenameHint');
  if (btnSetFn) {
    btnSetFn.replaceWith(btnSetFn.cloneNode(true));
  }
  const btnSetFn2 = document.getElementById('btnSetFilename');
  if (btnSetFn2) {
    btnSetFn2.addEventListener('click', () => {
      if (getMode() !== 'kis') {
        alert('Đổi tên file chỉ áp dụng cho KIS. (QA/Trake xuất mặc định)');
        return;
      }
      const cur = kis_fname_get();
      const name = prompt('Đặt tên file CSV (KIS):', cur);
      if (name !== null) {
        kis_fname_set(name);
        if (fnHint) fnHint.textContent = `(${getMode().toUpperCase()} mode) ${name}`;
      }
    });
  }

  // 4) Đồng bộ counter hiển thị theo mode
  syncAddedCounter();

  // 5) Bảo đảm nút phụ cho Trake
  ensureTrakeSetNButton();
  ensureTrakeAddLineButton();
}

/** Đồng bộ counter theo mode:
 *  - KIS  : số dòng đã Add trong kis_rows
 *  - QA   : số dòng đã Add trong qa_rows
 *  - Trake: số dòng đã chốt trong trakeRows
 */
function syncAddedCounter() {
  const mode = getMode();
  if (mode === 'kis') {
    setAddedCounter(Math.min(kis_rows_get().length, KIS_MAX_LINES));
  } else if (mode === 'qa') {
    setAddedCounter(Math.min(qa_rows_get().length, QA_MAX_LINES));
  } else if (mode === 'trake') {
    setAddedCounter(Math.min(tk_rows_get().length, TK_MAX_LINES));
  }
}

/* =========================================================================
 * (H) PLAYER / TIỆN ÍCH KHÁC
 * ========================================================================= */

/** Nút Add từ trang /watch (nếu có) -> thêm KIS */
(function initPlayerAdd() {
  const btn = document.getElementById('btnAddFromPlayer');
  if (!btn) return;
  const usp = new URLSearchParams(window.location.search);
  const vcode = usp.get('vcode') || '';
  const frame = parseInt(usp.get('frame') || '', 10);

  btn.addEventListener('click', () => {
    if (!vcode || Number.isNaN(frame)) return alert('Thiếu vcode/frame trong URL player.');
    if (!kis_addLine(vcode, frame)) alert('Đã đủ 100 dòng.');
  });
  syncAddedCounter();
})();

/** Mở trang /allframes?vcode=... trong tab mới (nếu cần) */
function goAllFrames(vcode) {
  const url = `/allframes?vcode=${encodeURIComponent(vcode)}`;
  window.open(url, '_blank');
}

/* =========================================================================
 * (I) KHỞI ĐỘNG
 * ========================================================================= */

document.addEventListener('DOMContentLoaded', () => {
  wireModeTabs();               // bắt 3 tab để đổi mode (không điều hướng)
  setMode(getMode() || 'kis');  // chọn mode ban đầu (mặc định KIS)
  syncAddedCounter();           // hiển thị Added theo mode
});

/* ======================================================================
 * [THÊM] Bảng “Nháp (Preview)” hiển thị các mục sẽ được tải về
 *  - Tự tạo một khối UI dưới phần Actions (panel phải)
 *  - Render khác nhau cho KIS / Q&A / TRAKE
 *  - Có nút xoá từng dòng & xoá tất cả
 * ====================================================================== */

/** (1) Tạo khung UI 1 lần nếu chưa có */
function ensureDraftPanelUI() {
  // Tìm panel phải (nơi có nút Submit)
  const submitBtn = document.getElementById('btnSubmitCSV');
  if (!submitBtn) return;
  const container = submitBtn.closest('aside') || submitBtn.parentElement;
  if (!container) return;

  // Nếu đã có khung thì thôi
  if (document.getElementById('draftPanel')) return;

  // Tạo khối “Nháp (Preview)”
  const wrap = document.createElement('div');
  wrap.id = 'draftPanel';
  wrap.className = 'mt-4 border rounded bg-white p-3';
  wrap.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <div class="text-sm font-semibold">Nháp (Preview)</div>
      <div class="text-xs text-slate-500" id="draftPanelModeHint"></div>
    </div>
    <div id="draftPanelBody" class="space-y-2 text-sm"></div>
    <div class="mt-2 flex items-center gap-2">
      <button id="btnDraftClear" class="px-2 py-1 rounded border hover:bg-slate-50 text-xs">Xoá tất cả</button>
    </div>
  `;
  // Chèn ngay phía dưới cụm nút Submit
  container.appendChild(wrap);

  // Gắn handler “Xoá tất cả”
  const btnClear = wrap.querySelector('#btnDraftClear');
  btnClear.addEventListener('click', () => {
    const mode = getMode();
    if (!confirm('Xoá toàn bộ nháp hiện tại?')) return;
    if (mode === 'kis') {
      localStorage.setItem('kis_rows', '[]');
    } else if (mode === 'qa') {
      localStorage.setItem('qa_rows', '[]');
    } else if (mode === 'trake') {
      localStorage.setItem('trakeRows', '[]'); // chỉ xoá danh sách dòng
      // Tùy bạn có muốn xoá draft luôn không; mặc định giữ draft để tiếp tục làm
      // localStorage.removeItem('trakeDraft');
    }
    renderDraftPanel();   // vẽ lại
    syncAddedCounter();   // đếm lại
  });

  // Delegation: Xoá 1 dòng (nếu có nút data-del-index)
  wrap.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-del-index]');
    if (!btn) return;
    const idx = parseInt(btn.getAttribute('data-del-index'), 10);
    const mode = getMode();
    if (Number.isNaN(idx)) return;

    if (mode === 'kis') {
      const rows = kis_rows_get();
      rows.splice(idx, 1);
      kis_rows_set(rows);
    } else if (mode === 'qa') {
      const rows = qa_rows_get();
      rows.splice(idx, 1);
      qa_rows_set(rows);
    } else if (mode === 'trake') {
      const rows = tk_rows_get();
      rows.splice(idx, 1);
      tk_rows_set(rows);
    }
    renderDraftPanel();
    syncAddedCounter();
  });
}

/** (2) Render bảng tuỳ theo mode */
function renderDraftPanel() {
  ensureDraftPanelUI();
  const body = document.getElementById('draftPanelBody');
  const hint = document.getElementById('draftPanelModeHint');
  if (!body || !hint) return;

  const mode = getMode();
  hint.textContent = `Chế độ: ${mode.toUpperCase()}`;

  // Xoá nội dung cũ
  body.innerHTML = '';

  if (mode === 'kis') {
    // KIS: danh sách {vcode, frame_idx}
    const rows = kis_rows_get();
    if (!rows.length) {
      body.innerHTML = `<div class="text-xs text-slate-500">Chưa có dòng nào. Hãy chọn frame rồi bấm Add.</div>`;
      return;
    }
    rows.forEach((r, i) => {
      const item = document.createElement('div');
      item.className = 'flex items-center justify-between border rounded px-2 py-1';
      item.innerHTML = `
        <div class="font-mono">${r.vcode}, ${r.frame_idx}</div>
        <button class="text-xs text-red-600 hover:underline" data-del-index="${i}">Xoá</button>
      `;
      body.appendChild(item);
    });
    // Gợi ý cách nộp
    const tip = document.createElement('div');
    tip.className = 'text-xs text-slate-500';
    tip.textContent = 'Khi Submit: hệ thống sẽ tự bổ sung từ grid để đủ 100 dòng nếu thiếu.';
    body.appendChild(tip);

  } else if (mode === 'qa') {
    // QA: danh sách {video, frame, answer}
    const rows = qa_rows_get();
    if (!rows.length) {
      body.innerHTML = `<div class="text-xs text-slate-500">Chưa có dòng nào. Bấm Add để thêm Answer cho frame đã chọn.</div>`;
      return;
    }
    rows.forEach((r, i) => {
      const item = document.createElement('div');
      item.className = 'border rounded px-2 py-1';
      item.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="font-mono">${r.video}, ${r.frame}</div>
          <button class="text-xs text-red-600 hover:underline" data-del-index="${i}">Xoá</button>
        </div>
        <div class="text-xs italic break-words">"${r.answer}"</div>
      `;
      body.appendChild(item);
    });
    const tip = document.createElement('div');
    tip.className = 'text-xs text-slate-500';
    tip.textContent = 'Submit sẽ tải file qa_submission.csv gồm tất cả dòng ở trên.';
    body.appendChild(tip);

  } else if (mode === 'trake') {
    // TRAKE: hiển thị draft hiện tại + danh sách rows đã chốt
    const draft = tk_draft_get();
    // 2.1 Draft hiện tại
    const draftBox = document.createElement('div');
    draftBox.className = 'border rounded p-2 bg-slate-50';
    if (!draft) {
      draftBox.innerHTML = `<div class="text-xs text-slate-500">Chưa khởi tạo N. Hãy bấm "Set N".</div>`;
    } else {
      const { video, N, frames } = draft;
      const lines = [];
      for (let i = 0; i < N; i++) {
        const val = (frames && frames[i] != null) ? frames[i] : '—';
        lines.push(`<span class="px-1 py-0.5 rounded border">${i+1}:${val}</span>`);
      }
      draftBox.innerHTML = `
        <div class="text-xs mb-1">Draft hiện tại:</div>
        <div class="text-xs"><b>Video:</b> <span class="font-mono">${video || '(chưa có)'}</span> &nbsp; <b>N:</b> ${N}</div>
        <div class="mt-1 flex flex-wrap gap-1 text-xs">${lines.join(' ')}</div>
        <div class="mt-1 text-[11px] text-slate-500">Bấm Add để gán frame vào vị trí (sẽ hỏi “hành động thứ mấy?”).</div>
      `;
    }
    body.appendChild(draftBox);

    // 2.2 Danh sách dòng đã chốt
    const rows = tk_rows_get();
    const head = document.createElement('div');
    head.className = 'mt-2 text-xs text-slate-500';
    head.textContent = `Đã chốt: ${rows.length} dòng`;
    body.appendChild(head);

    if (!rows.length) {
      const empty = document.createElement('div');
      empty.className = 'text-xs text-slate-500';
      empty.textContent = 'Chưa có dòng nào. Khi draft đủ N frame, bấm "Add Line" để chốt.';
      body.appendChild(empty);
    } else {
      rows.forEach((r, i) => {
        const item = document.createElement('div');
        item.className = 'border rounded px-2 py-1';
        const seq = r.frames.map((f, idx) => `<span class="px-1 py-0.5 rounded border">${idx+1}:${f}</span>`).join(' ');
        item.innerHTML = `
          <div class="flex items-center justify-between">
            <div class="text-xs"><b class="font-mono">${r.video}</b> &nbsp; <span class="text-slate-500">N=${r.N}</span></div>
            <button class="text-xs text-red-600 hover:underline" data-del-index="${i}">Xoá</button>
          </div>
          <div class="mt-1 flex flex-wrap gap-1 text-xs">${seq}</div>
        `;
        body.appendChild(item);
      });

      const tip = document.createElement('div');
      tip.className = 'text-xs text-slate-500';
      tip.textContent = 'Submit sẽ tải file trake_submission.csv gồm TẤT CẢ các dòng đã chốt.';
      body.appendChild(tip);
    }
  }
}



/** (3) Kết nối vòng đời: vẽ lại khi:
 *  - DOM sẵn sàng
 *  - đổi mode (rebindActionsForMode đã có)
 *  - sau khi Add / Add Line / Clear / Delete
 */
document.addEventListener('DOMContentLoaded', () => {
  ensureDraftPanelUI();
  renderDraftPanel();
});

// Nếu có hàm rebindActionsForMode (đã dùng để gắn Add/Submit theo mode), bọc lại để render
if (typeof window.rebindActionsForMode === 'function') {
  const __oldRebindDP = window.rebindActionsForMode;
  window.rebindActionsForMode = function(){
    __oldRebindDP();
    ensureDraftPanelUI();
    renderDraftPanel();
  };
}

// Sau mỗi lần bấm Add / Add Line, chúng ta đã gọi set vào localStorage → gọi renderDraftPanel()
// Dưới đây “bọc nhẹ” các setter để auto-render.

// KIS
if (typeof window.kis_rows_set === 'function' && !window.__wrap_kis_rows_set_for_preview) {
  const _old = window.kis_rows_set;
  window.kis_rows_set = function(a){ _old(a); renderDraftPanel(); };
  window.__wrap_kis_rows_set_for_preview = true;
}
// QA
if (typeof window.qa_rows_set === 'function' && !window.__wrap_qa_rows_set_for_preview) {
  const _old = window.qa_rows_set;
  window.qa_rows_set = function(a){ _old(a); renderDraftPanel(); };
  window.__wrap_qa_rows_set_for_preview = true;
}
// TRAKE rows
if (typeof window.tk_rows_set === 'function' && !window.__wrap_tk_rows_set_for_preview) {
  const _old = window.tk_rows_set;
  window.tk_rows_set = function(a){ _old(a); renderDraftPanel(); };
  window.__wrap_tk_rows_set_for_preview = true;
}
// TRAKE draft
if (typeof window.tk_draft_set === 'function' && !window.__wrap_tk_draft_set_for_preview) {
  const _old = window.tk_draft_set;
  window.tk_draft_set = function(s){ _old(s); renderDraftPanel(); };
  window.__wrap_tk_draft_set_for_preview = true;
}
(function enhanceAddFileButton(){
  const btn = document.getElementById('btnSetFilename');
  if (!btn) return;
  const hint = document.getElementById('csvFilenameHint');

  btn.addEventListener('click', () => {
    const mode = getMode();
    let suggest = '';

    if (mode === 'kis')   suggest = 'query-1-kis.csv';
    if (mode === 'qa')    suggest = 'query-2-qa.csv';
    if (mode === 'trake') suggest = 'query-3-trake.csv';

    const cur = (mode === 'kis') ? (kis_fname_get() || suggest) : suggest;
    const msg = `Đặt tên file CSV (${mode.toUpperCase()}):\n\nGợi ý:\n- KIS   → query-1-kis.csv\n- Q&A   → query-2-qa.csv\n- Trake → query-3-trake.csv\n\nBạn có thể giữ gợi ý hoặc nhập tên khác:`;
    const name = prompt(msg, cur);

    if (name !== null && name.trim()) {
      if (mode === 'kis') {
        kis_fname_set(name.trim());
      } else {
        // QA/Trake không có hàm đặt tên riêng -> chỉ hiển thị hint
        localStorage.setItem(mode + '_fname', name.trim());
      }
      if (hint) hint.textContent = `(${mode.toUpperCase()} mode) ${name.trim()}`;
    }
  });
})();

/* ======================================================================
 * PATCH: Dùng tên file đã chọn cho cả QA & TRAKE khi Submit
 * ====================================================================== */

function qa_fname_get() { return localStorage.getItem('qa_fname') || 'query-2-qa.csv'; }
function trake_fname_get() { return localStorage.getItem('trake_fname') || 'query-3-trake.csv'; }

// Ghi đè nhẹ submit QA để dùng tên file
if (typeof qa_submit === 'function') {
  const __old_qa_submit = qa_submit;
  qa_submit = function() {
    const rows = qa_rows_get();
    if (!rows.length) return alert('Danh sách trống. Hãy Add ít nhất 1 dòng.');
    const lines = rows.map(it => `${it.video}, ${it.frame}, "${it.answer}"`).join('\n');
    const fn = qa_fname_get();
    const blob = new Blob([lines], { type:'text/csv;charset=utf-8;' });
    const a = Object.assign(document.createElement('a'), { href:URL.createObjectURL(blob), download: fn });
    a.click();
  };
}

// Ghi đè nhẹ submit Trake để dùng tên file
if (typeof tk_submit_all === 'function') {
  const __old_tk_submit_all = tk_submit_all;
  tk_submit_all = function() {
    const rows = tk_rows_get();
    const s = tk_draft_get();
    if (s && s.video && Array.isArray(s.frames) && !s.frames.some(x => x==null)) {
      const addNow = confirm('Draft hiện tại đã đủ frame. Thêm vào danh sách trước khi Submit không?');
      if (addNow) { rows.push({ video:s.video, frames:s.frames.slice(), N:s.N }); }
    }
    if (!rows.length) return alert('Chưa có dòng nào. Hãy bấm "Add Line" sau khi gán đủ N frame.');
    const lines = rows.map(r => [r.video, ...r.frames].join(', ')).join('\n');
    const fn = trake_fname_get();
    const blob = new Blob([lines], { type:'text/csv;charset=utf-8;' });
    const a = Object.assign(document.createElement('a'), { href:URL.createObjectURL(blob), download: fn });
    a.click();
  };
}
