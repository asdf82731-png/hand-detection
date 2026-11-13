// Past results, kept only on this phone. Storage can be unavailable (private
// mode, blocked site data); the app works without it.
const KEY = 'stroke-counter-history';
const MAX = 20;

export function loadHistory() {
  try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch { return []; }
}

export function saveHistory(name, data) {
  const entry = {at: new Date().toISOString(), name, left: data.left_hand_counts || {}, right: data.right_hand_counts || {}};
  const history = [entry, ...loadHistory()].slice(0, MAX);
  try { localStorage.setItem(KEY, JSON.stringify(history)); } catch { /* storage unavailable */ }
  return history;
}

export function clearHistory() {
  try { localStorage.removeItem(KEY); } catch { /* ignore */ }
  return [];
}
