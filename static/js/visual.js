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

  window.open(url.toString(), '_blank');   // mở tab mới giống Visione
});
