/**
 * AttendAI — Shared Theme & UI Utilities
 * Include this script in every page for consistent theme toggle and icon behavior.
 */

// ── Theme Management ──
(function initTheme() {
  const saved = localStorage.getItem('attendai-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
})();

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('attendai-theme', next);
  
  // Update toggle button icon
  const btn = document.getElementById('themeToggle');
  if (btn) {
    btn.querySelector('.toggle-icon').textContent = next === 'dark' ? '🌙' : '☀️';
    btn.querySelector('.toggle-label').textContent = next === 'dark' ? 'Dark Mode' : 'Light Mode';
  }
}

// ── Auto-inject theme toggle into sidebar ──
document.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.querySelector('.sidebar');
  if (sidebar) {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const toggleDiv = document.createElement('div');
    toggleDiv.className = 'theme-toggle';
    toggleDiv.id = 'themeToggle';
    toggleDiv.onclick = toggleTheme;
    toggleDiv.innerHTML = `
      <span class="toggle-icon">${current === 'dark' ? '🌙' : '☀️'}</span>
      <span class="toggle-label">${current === 'dark' ? 'Dark Mode' : 'Light Mode'}</span>
    `;
    // Insert before the sign-out button
    const signOut = sidebar.querySelector('[onclick*="logout"]');
    if (signOut) {
      signOut.parentElement.insertBefore(toggleDiv, signOut);
    } else {
      sidebar.appendChild(toggleDiv);
    }
  }

  // Initialize Lucide icons if loaded
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
});
