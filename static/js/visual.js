// Lấy form search để giữ tham số UI hiện tại
const searchForm = document.querySelector('form[action*="textsearch"]');

// Helper: lấy value theo name trong form
const getVal = (name) =>
  (searchForm?.querySelector(`input[name="${name}"]`)?.value ?? '').trim();

// Hàm chọn card ảnh: gắn vào window để HTML gọi được
window.onPick = function (card) {
  // --- Lấy data từ card ---
  const id    = card.getAttribute('data-id');
  const path  = card.getAttribute('data-path');
  const score = card.getAttribute('data-score');

  // --- Tham chiếu panel ---
  const ph   = document.getElementById('pvPh');
  const img  = document.getElementById('pvImg');
  const vid  = document.getElementById('pvVid');
  const idEl = document.getElementById('pvId');
  const scEl = document.getElementById('pvScore');
  const tgBtn= document.getElementById('btnToggle');
  const sim  = document.getElementById('btnSimilar');

  // --- Cập nhật ảnh preview ---
  img.src = "/get_img?fpath=" + encodeURIComponent(path);
  img.classList.remove('hidden');
  vid.classList.add('hidden');
  ph.classList.add('hidden');
  tgBtn.disabled = false;

  // --- Điền thông tin ---
  idEl.textContent = id;
  scEl.textContent = Number(score || 0).toFixed(4);

  // --- Build URL Find similar (giữ tham số UI) ---
  const params = new URLSearchParams({
    imgid: id,
    index: 0,
    textquery: getVal('textquery'),
    search_type: getVal('search_type') || 'visual',
    topk: getVal('topk') || '100',
    wvis: getVal('wvis') || '0.6',
    wtxt: getVal('wtxt') || '0.4',
    constraints: getVal('constraints') || ''
  });
  sim.href = "/imgsearch?" + params.toString();
  sim.removeAttribute('aria-disabled');
};

// (tuỳ chọn) Toggle media khi sau này bạn set được video src cho frame
document.getElementById('btnToggle')?.addEventListener('click', function () {
  const img = document.getElementById('pvImg');
  const vid = document.getElementById('pvVid');
  if (!vid.src) return; // chưa có video thì bỏ qua
  const showingImg = !img.classList.contains('hidden');
  if (showingImg) {
    img.classList.add('hidden');
    vid.classList.remove('hidden');
    this.textContent = 'Show image';
  } else {
    vid.classList.add('hidden');
    img.classList.remove('hidden');
    this.textContent = 'Show video';
  }
});
