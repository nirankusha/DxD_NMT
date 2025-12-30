fetch('/DxD_NMT/data/ENArticles_PLTranslation.csv')
  .then(r => r.text())
  .then(text => {
    const rows = text.trim().split('\n').map(r => r.split(','));
    const header = rows[0];

    const select = document.getElementById('columns');
    header.forEach((h, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = h;
      select.appendChild(opt);
    });

    select.addEventListener('change', () => {
      const selected = [...select.selectedOptions].map(o => +o.value);
      const table = document.getElementById('table');
      table.innerHTML = '';

      rows.forEach((row, rIdx) => {
        const tr = document.createElement('tr');
        selected.forEach(i => {
          const cell = document.createElement(rIdx === 0 ? 'th' : 'td');
          cell.textContent = row[i] ?? '';
          tr.appendChild(cell);
        });
        table.appendChild(tr);
      });
    });
  });

