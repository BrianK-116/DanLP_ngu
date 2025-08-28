/* =========================
 * visual.js — preview & actions (phiên bản dùng n từ tên file)
 * - Click keyframe => cập nhật panel phải (thumbnail + info)
 * - Find similar  => điều hướng /imgsearch?imgid=... (giữ tham số UI hiện tại)
 * - Show video    => mở tab mới /watch?vcode=<Lxx_Vyyy>&frame=<n>
 *   (backend sẽ dùng load_pts_map để tua đúng pts_time)
 * ========================= */

/** Form tìm kiếm để tái dùng tham số UI hiện có */
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
  const m = fname.match(/(\d+)/);     // bắt dãy số cuối tên file
  return m ? parseInt(m[1], 10) : 0;  // fallback 0 nếu không bắt được
}

/** Handler khi chọn 1 thẻ keyframe trong kết quả */
window.onPick = function (card) {
  // --- lấy data từ item ---
  const id    = card.getAttribute('data-id');     // id nội bộ (chỉ để hiển thị)
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
    imgid: id,                                        // server của bạn đang dùng id nội bộ cho /imgsearch
    index: 0,
    textquery: getVal('textquery'),
    search_type: getVal('search_type') || 'visual',   // chỉ visual/ocr
    topk: getVal('topk') || '100'
  });
  sim.href = '/imgsearch?' + params.toString();
  sim.removeAttribute('aria-disabled');

  // --- chuẩn bị thông tin để mở video ở tab mới ---
  const vcode = extractVcode(path);   // "L21_V001"
  const n     = extractN(path);       // số nguyên từ tên file, ví dụ 20
  tgBtn.dataset.vcode = vcode;
  tgBtn.dataset.frame = String(n);    // 'frame' ở URL chính là n
};

/** Nút Show video -> mở tab mới tới /watch?vcode=<vcode>&frame=<n>
 *  Backend sẽ dùng load_pts_map để tra pts_time theo n và tua đúng mốc.
 */
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


const SUBM_KIND_KEY = 'subm_kind';      // loại nộp (mặc định 'kis')
const SUBM_ROWS_KEY = 'subm_rows';      // danh sách dòng đã add
const SUBM_FNAME_KEY = 'subm_fname';    // tên file CSV mong muốn
const SUBM_MAX_LINES = 100;             // tối đa 100 dòng

// ---------- Tiện ích localStorage ----------
function getRows(){ try { return JSON.parse(localStorage.getItem(SUBM_ROWS_KEY) || '[]'); } catch { return []; } }
function setRows(arr){ localStorage.setItem(SUBM_ROWS_KEY, JSON.stringify(arr || [])); }
function getFname(){ return localStorage.getItem(SUBM_FNAME_KEY) || ''; }
function setFname(n){ localStorage.setItem(SUBM_FNAME_KEY, String(n||'').trim()); }
function updateCounters(){
  const n = Math.min(getRows().length, SUBM_MAX_LINES);
  const el1 = document.getElementById('addedCount');
  const el2 = document.getElementById('addedCountPlayer');
  if (el1) el1.textContent = n;
  if (el2) el2.textContent = n;
}

// ---------- Lưu lựa chọn hiện tại (Preview) ----------
/** [THÊM] Trạng thái chọn hiện hành trong Preview */
const Sel = { vcode: null, frame_idx: null };

/** [THÊM] Ưu tiên lấy vcode & frame từ IMG có data-* */
function getVcodeFrameFromImgEl(imgEl){
  // 1) Ưu tiên data-frame (n) + path để suy ra vcode
  const df = imgEl.getAttribute('data-frame'); // kỳ vọng là 'n'
  let n = (df!=null) ? parseInt(df, 10) : NaN;

  // 2) Lấy vcode từ data-path nếu có
  const p = imgEl.getAttribute('data-path') || '';
  let vcode = null;
  if (p){
    const parts = p.split('/').filter(Boolean);
    // .../Keyframes_L21/L21_V001/020.jpg -> parts[-2] = L21_V001
    vcode = parts[parts.length - 2] || null;
    // Nếu chưa có n (data-frame thiếu), suy từ tên file ảnh
    if (Number.isNaN(n)){
      const fname = parts[parts.length - 1] || '';
      const num = parseInt(fname, 10);
      if (!Number.isNaN(num)) n = num;
    }
  }
  if (vcode && Number.isInteger(n) && n >= 0) return { vcode, frame_idx: n };
  return null;
}

/** [THÊM] Thêm một hàng KIS */
function addKISRow(vcode, frameIdx){
  const rows = getRows();
  if (rows.length >= SUBM_MAX_LINES) return false;
  rows.push({ vcode, frame_idx: Number(frameIdx) });
  setRows(rows);
  updateCounters();
  return true;
}

/** [THÊM] Tải file CSV từ server (/export/kis), server sẽ pad đủ 100 */
async function downloadKISCsv(filename){
  const rows = getRows();
  const fn = filename || getFname() || 'submit_kis.csv';
  const res = await fetch('/export/kis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows, filename: fn })
  });
  if (!res.ok){
    const t = await res.text();
    alert('Export failed: ' + t);
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = fn;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

/* =========================
 * GẮN SỰ KIỆN Ở TRANG HOME
 * ========================= */
(function initHomeButtonsV2(){
  // Khi click một IMG kết quả -> cập nhật lựa chọn hiện hành & bật nút Add
  document.addEventListener('click', (ev) => {
    const el = ev.target;
    if (!el || el.tagName !== 'IMG') return;
    if (!el.hasAttribute('data-path')) return; // chỉ xử lý IMG kết quả

    const parsed = getVcodeFrameFromImgEl(el);
    const btnAdd = document.getElementById('btnAddFromPreview');
    if (parsed){
      Sel.vcode = parsed.vcode;
      Sel.frame_idx = parsed.frame_idx;
      if (btnAdd) btnAdd.disabled = false; // bật nút Add khi đã có lựa chọn hợp lệ
    }
  });

  // Hành vi của nút Add trong Preview
  const btnAdd = document.getElementById('btnAddFromPreview');
  if (btnAdd){
    btnAdd.addEventListener('click', () => {
      if (!Sel.vcode || !Number.isInteger(Sel.frame_idx)){
        alert('Chưa chọn khung hình hợp lệ.');
        return;
      }
      const ok = addKISRow(Sel.vcode, Sel.frame_idx);
      if (!ok) alert('Đã đủ 100 dòng.');
    });
  }

  // Nút đặt tên file (Add File)
  const btnSetFn = document.getElementById('btnSetFilename');
  const fnHint = document.getElementById('csvFilenameHint');
  if (btnSetFn){
    const show = () => { if (fnHint) fnHint.textContent = getFname(); };
    show();
    btnSetFn.addEventListener('click', () => {
      const cur = getFname() || 'submit_kis.csv';
      const name = prompt('Đặt tên file CSV:', cur);
      if (name !== null){ setFname(name); show(); }
    });
  }

  // Nút Submit
  const btnSubmit = document.getElementById('btnSubmitCSV');
  if (btnSubmit){
    btnSubmit.addEventListener('click', () => {
      downloadKISCsv().catch(err => alert(err));
    });
  }

  updateCounters();
})();

/* =========================
 * GẮN SỰ KIỆN Ở TRANG PLAYER
 * ========================= */
(function initPlayerButtonsV2(){
  const btnAddP = document.getElementById('btnAddFromPlayer');
  if (!btnAddP) return;

  // Lấy vcode & frame từ URL (/watch?vcode=...&frame=...)
  const usp = new URLSearchParams(window.location.search);
  const vcode = usp.get('vcode') || '';
  const frame = parseInt(usp.get('frame') || '', 10);

  btnAddP.addEventListener('click', () => {
    if (!vcode || Number.isNaN(frame)){
      alert('Thiếu vcode/frame trong URL player.');
      return;
    }
    const ok = addKISRow(vcode, frame);
    if (!ok) alert('Đã đủ 100 dòng.');
  });

  updateCounters();
})();
/* ==========================================================
 * [THÊM] Bắt click trên .card (đúng nơi có data-id/data-path/data-frame)
 * - Lưu lựa chọn hiện hành Sel = { vcode, frame_idx }
 * - Bật nút Add và đổi màu (xanh đậm) cho rõ trạng thái
 * ========================================================== */
(function wireSelectFromCard(){
  // Biến trạng thái chọn hiện hành (tái sử dụng nếu đã khai báo trước)
  window.Sel = window.Sel || { vcode: null, frame_idx: null };

  document.addEventListener('click', (ev) => {
    // Tìm phần tử .card gần nhất mà ta vừa click (thumbnail wrapper)
    const card = ev.target?.closest?.('.card');
    if (!card) return;

    // Lấy dữ liệu từ data-* TRÊN .card (đúng theo home.html)
    const path  = card.getAttribute('data-path')  || '';
    const frame = card.getAttribute('data-frame');        // n
    const parts = path.split('/').filter(Boolean);
    const vcode = parts.length >= 2 ? parts[parts.length - 2] : null;
    const n = frame != null ? parseInt(frame, 10) : NaN;

    if (!vcode || Number.isNaN(n) || n < 0) return;

    // Lưu lựa chọn hiện hành
    Sel.vcode = vcode;
    Sel.frame_idx = n;

    // Bật nút Add & đổi màu cho rõ
    const btnAdd = document.getElementById('btnAddFromPreview');
    if (btnAdd){
      btnAdd.disabled = false;
      // nếu trước đó đang xám, chuyển sang xanh đậm
      btnAdd.classList.remove('bg-gray-400');
      btnAdd.classList.add('bg-emerald-600','hover:bg-emerald-700','text-white');
    }
  });
})();
/* ==========================================================
 * [THÊM] Khởi tạo đồng hồ đếm khi trang sẵn sàng
 * ========================================================== */
document.addEventListener('DOMContentLoaded', () => {
  if (typeof updateCounters === 'function') {
    updateCounters(); // cập nhật Added: X/100 ngay khi load
  }
});
/* ==========================================================
 * [THÊM] An toàn: nếu sau khi Add bạn chưa gọi updateCounters()
 * thì khối này sẽ gắn vào nút Add để cập nhật ngay.
 * (Không trùng lặp; nếu bạn đã gọi ở nơi khác thì cũng ok)
 * ========================================================== */
(function ensureCounterAfterAdd(){
  const btnAdd = document.getElementById('btnAddFromPreview');
  if (!btnAdd) return;
  btnAdd.addEventListener('click', () => {
    // gọi trễ 1 tick để đảm bảo localStorage đã được cập nhật
    setTimeout(() => {
      if (typeof updateCounters === 'function') updateCounters();
    }, 0);
  }, { capture: false });
})();

function __collectGridFrames() {
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

/** # Hợp nhất: giữ thứ tự đã Add, sau đó bổ sung từ grid cho đủ 100 (không trùng) */
function __buildFinalRowsFromAddAndGrid() {
  const MAX = 100;

  // 1) Lấy danh sách đã Add (giữ thứ tự)
  const added = getRows(); // [{vcode, frame_idx}, ...] — từ localStorage

  // 2) Lấy danh sách grid theo DOM hiện tại
  const grid = __collectGridFrames();

  // 3) Tạo tập kiểm trùng (khóa 'vcode#frame')
  const seen = new Set();
  const key = (r) => `${r.vcode}#${r.frame_idx}`;

  const finalRows = [];
  // 3.1) Đưa phần đã Add lên trước theo đúng thứ tự Add
  for (const r of added) {
    const k = key(r);
    if (!seen.has(k)) {
      finalRows.push(r);
      seen.add(k);
      if (finalRows.length >= MAX) return finalRows;
    }
  }

  // 3.2) Bổ sung từ grid theo thứ tự hiển thị, bỏ qua cái đã có
  for (const r of grid) {
    const k = key(r);
    if (!seen.has(k)) {
      finalRows.push(r);
      seen.add(k);
      if (finalRows.length >= MAX) return finalRows;
    }
  }

  // 3.3) Nếu vẫn chưa đủ (hiếm), lặp lại phần tử cuối cùng
  if (finalRows.length > 0) {
    while (finalRows.length < MAX) finalRows.push(finalRows[finalRows.length - 1]);
  }
  return finalRows;
}

/** # Gửi CSV + dọn bộ nhớ */
async function __submitKIS_usingGrid() {
  // 1) Xây 100 dòng theo yêu cầu
  const rows100 = __buildFinalRowsFromAddAndGrid();

  // 2) Lấy tên file (nếu có)
  const fn = (typeof getFname === 'function' ? (getFname() || 'submit_kis.csv') : 'submit_kis.csv');

  // 3) Gửi lên server (vẫn /export/kis như cũ)
  const res = await fetch('/export/kis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows: rows100, filename: fn })
  });

  if (!res.ok) {
    const t = await res.text();
    alert('Export failed: ' + t);
    return;
  }

  // 4) Tải file
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = fn;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);

  // 5) XÓA BỘ NHỚ TẠM + cập nhật đồng hồ + (tùy) disable nút Add
  try {
    setRows([]);                 // xóa danh sách đã Add
    if (typeof updateCounters === 'function') updateCounters();
    const btnAdd = document.getElementById('btnAddFromPreview');
    if (btnAdd) btnAdd.disabled = true;  // sau submit, tắt Add tới khi chọn lại
  } catch (e) {
    console.warn('cleanup after submit failed:', e);
  }
}

/** # Bắt nút Submit ở pha CAPTURE để chặn handler cũ và dùng logic mới
 *   Không cần sửa code cũ: listener này chạy trước và stopImmediatePropagation()
 */
(function hijackSubmitButton(){
  const btnSubmit = document.getElementById('btnSubmitCSV');
  if (!btnSubmit) return;

  btnSubmit.addEventListener('click', (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    ev.stopImmediatePropagation(); // chặn các listener đã gắn trước đó
    __submitKIS_usingGrid().catch(err => alert(err));
  }, { capture: true }); // chạy sớm ở pha capture để đảm bảo chặn được
})();function __collectGridFrames() {
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

/** # Hợp nhất: giữ thứ tự đã Add, sau đó bổ sung từ grid cho đủ 100 (không trùng) */
function __buildFinalRowsFromAddAndGrid() {
  const MAX = 100;

  // 1) Lấy danh sách đã Add (giữ thứ tự)
  const added = getRows(); // [{vcode, frame_idx}, ...] — từ localStorage

  // 2) Lấy danh sách grid theo DOM hiện tại
  const grid = __collectGridFrames();

  // 3) Tạo tập kiểm trùng (khóa 'vcode#frame')
  const seen = new Set();
  const key = (r) => `${r.vcode}#${r.frame_idx}`;

  const finalRows = [];
  // 3.1) Đưa phần đã Add lên trước theo đúng thứ tự Add
  for (const r of added) {
    const k = key(r);
    if (!seen.has(k)) {
      finalRows.push(r);
      seen.add(k);
      if (finalRows.length >= MAX) return finalRows;
    }
  }

  // 3.2) Bổ sung từ grid theo thứ tự hiển thị, bỏ qua cái đã có
  for (const r of grid) {
    const k = key(r);
    if (!seen.has(k)) {
      finalRows.push(r);
      seen.add(k);
      if (finalRows.length >= MAX) return finalRows;
    }
  }

  // 3.3) Nếu vẫn chưa đủ (hiếm), lặp lại phần tử cuối cùng
  if (finalRows.length > 0) {
    while (finalRows.length < MAX) finalRows.push(finalRows[finalRows.length - 1]);
  }
  return finalRows;
}

/** # Gửi CSV + dọn bộ nhớ */
async function __submitKIS_usingGrid() {
  // 1) Xây 100 dòng theo yêu cầu
  const rows100 = __buildFinalRowsFromAddAndGrid();

  // 2) Lấy tên file (nếu có)
  const fn = (typeof getFname === 'function' ? (getFname() || 'submit_kis.csv') : 'submit_kis.csv');

  // 3) Gửi lên server (vẫn /export/kis như cũ)
  const res = await fetch('/export/kis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows: rows100, filename: fn })
  });

  if (!res.ok) {
    const t = await res.text();
    alert('Export failed: ' + t);
    return;
  }

  // 4) Tải file
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = fn;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);

  // 5) XÓA BỘ NHỚ TẠM + cập nhật đồng hồ + (tùy) disable nút Add
  try {
    setRows([]);                 // xóa danh sách đã Add
    if (typeof updateCounters === 'function') updateCounters();
    const btnAdd = document.getElementById('btnAddFromPreview');
    if (btnAdd) btnAdd.disabled = true;  // sau submit, tắt Add tới khi chọn lại
  } catch (e) {
    console.warn('cleanup after submit failed:', e);
  }
}

/** # Bắt nút Submit ở pha CAPTURE để chặn handler cũ và dùng logic mới
 *   Không cần sửa code cũ: listener này chạy trước và stopImmediatePropagation()
 */
(function hijackSubmitButton(){
  const btnSubmit = document.getElementById('btnSubmitCSV');
  if (!btnSubmit) return;

  btnSubmit.addEventListener('click', (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    ev.stopImmediatePropagation(); // chặn các listener đã gắn trước đó
    __submitKIS_usingGrid().catch(err => alert(err));
  }, { capture: true }); // chạy sớm ở pha capture để đảm bảo chặn được
})();

// (G) HÀM MỚI: điều hướng tới trang allframes
function goAllFrames(vcode) {
  // Xây URL /allframes?vcode=...
  const url = `/allframes?vcode=${encodeURIComponent(vcode)}`;
  // Mở tab mới hoặc điều hướng thẳng (tuỳ ý)
  window.open(url, '_blank'); // mở tab mới để tiện so sánh
}

