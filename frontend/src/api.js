import {HANDS, STROKES} from './strokes.js';

const POLL_MS = 600;
// Share of the progress bar given to the upload; the server's analysis gets the rest.
const UPLOAD_SHARE = 0.3;

/** {"left": {"Full": 10}, ...} from the target inputs, as the server expects; "" when none are filled in. */
export function expectedCountsJson(targets) {
  const expected = {};
  for (const [hand, handName] of HANDS) {
    for (const [stroke, name] of STROKES) {
      const count = targets[hand]?.[stroke];
      if (count == null || Number.isNaN(count)) continue;
      if (!Number.isInteger(count) || count < 0) throw new Error(`${handName}${name}：請填 0 以上的整數`);
      if (count > 0) (expected[hand] ||= {})[stroke] = count;
    }
  }
  return Object.keys(expected).length ? JSON.stringify(expected) : '';
}

function upload(file, expected, onProgress) {
  // XHR rather than fetch, for upload progress on slow phone connections.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('video_file', file, file.name || 'video.mp4');
    form.append('expected_counts', expected);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/jobs');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress({fraction: UPLOAD_SHARE * e.loaded / e.total, text: `上傳中… ${Math.round(100 * e.loaded / e.total)}%`});
    };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `上傳失敗（HTTP ${xhr.status}）`));
    };
    xhr.onerror = () => reject(new Error('連不上伺服器，請確認網路連線'));
    xhr.send(form);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Upload, then poll the background job until it is done. Returns the result fields. */
export async function analyzeVideo(file, expected, onProgress) {
  onProgress({fraction: 0, text: '上傳中…'});
  const {job_id: jobId} = await upload(file, expected, onProgress);
  for (;;) {
    await sleep(POLL_MS);
    const response = await fetch(`/jobs/${jobId}`, {cache: 'no-store'});
    if (!response.ok) throw new Error(`找不到這次的分析（HTTP ${response.status}），請重新上傳`);
    const job = await response.json();
    if (job.status === 'done') { onProgress({fraction: 1, text: '完成'}); return job.result; }
    if (job.status === 'error') throw new Error(job.detail);
    onProgress({fraction: UPLOAD_SHARE + (1 - UPLOAD_SHARE) * job.progress, text: `分析中… ${Math.round(job.progress * 100)}%`});
  }
}

const STROKE_NAME = Object.fromEntries(STROKES.map(([key, name]) => [key, name]));
const handName = (hand) => (hand === 'left' ? '左手' : '右手');

/** The server writes suggestions in English; show them in Chinese. */
export function translateSuggestion(text) {
  let m = text.match(/^Try making the (left|right) hand's '(\w+)' gesture more distinct — it reached (\d+)%, below (\d+)%\.$/);
  if (m) return `${handName(m[1])}的${STROKE_NAME[m[2]] || m[2]}只達 ${m[3]}%（低於 ${m[4]}%），動作可以再做得更明確。`;
  m = text.match(/^The (left|right) hand's '(\w+)' strokes were stable \((\d+)%\) — keep it up\.$/);
  if (m) return `${handName(m[1])}的${STROKE_NAME[m[2]] || m[2]}很穩定（${m[3]}%），繼續保持。`;
  return text;
}
