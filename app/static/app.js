document.querySelectorAll('.password-toggle').forEach((button) => {
  button.addEventListener('click', () => {
    const input = button.parentElement.querySelector('input');
    const visible = input.type === 'text';
    input.type = visible ? 'password' : 'text';
    button.textContent = visible ? 'Show' : 'Hide';
  });
});

const sourceSwitch = document.querySelector('[data-source-switch]');
if (sourceSwitch) {
  const syncSource = () => {
    const selected = sourceSwitch.querySelector('input:checked').value;
    document.querySelectorAll('[data-source-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.sourcePanel !== selected;
    });
  };
  sourceSwitch.addEventListener('change', syncSource);
  syncSource();
}

const fileInput = document.querySelector('[data-csv-input]');
if (fileInput) {
  const dropzone = document.querySelector('[data-dropzone]');
  const preview = document.querySelector('[data-csv-preview]');
  const tabs = document.querySelector('[data-file-tabs]');
  const table = document.querySelector('[data-preview-table]');
  const summary = document.querySelector('[data-preview-summary]');
  const filenameLabel = document.querySelector('[data-file-name]');
  const addButton = document.querySelector('[data-add-csv]');
  const selectedFiles = new Map();
  let activeKey = null;

  const fileKey = (file) => `${file.name}:${file.size}:${file.lastModified}`;

  const syncInputFiles = () => {
    const transfer = new DataTransfer();
    selectedFiles.forEach((file) => transfer.items.add(file));
    fileInput.files = transfer.files;
  };

  const parseCsv = (text) => {
    const rows = [];
    let row = [];
    let value = '';
    let quoted = false;
    for (let index = 0; index < text.length; index += 1) {
      const char = text[index];
      if (quoted) {
        if (char === '"' && text[index + 1] === '"') { value += '"'; index += 1; }
        else if (char === '"') quoted = false;
        else value += char;
      } else if (char === '"') quoted = true;
      else if (char === ',') { row.push(value); value = ''; }
      else if (char === '\n') { row.push(value.replace(/\r$/, '')); rows.push(row); row = []; value = ''; }
      else value += char;
    }
    if (value.length || row.length) { row.push(value.replace(/\r$/, '')); rows.push(row); }
    return rows.filter((item) => item.some((cell) => cell.trim() !== ''));
  };

  const renderTable = async (file) => {
    table.innerHTML = '<tbody><tr><td class="preview-loading">Reading preview…</td></tr></tbody>';
    try {
      const rows = parseCsv(await file.text());
      if (!rows.length) throw new Error('This CSV is empty.');
      const headers = rows[0];
      const shown = rows.slice(1, 13);
      const head = document.createElement('thead');
      const headRow = document.createElement('tr');
      headers.forEach((header) => {
        const cell = document.createElement('th');
        cell.textContent = header || 'Unnamed';
        headRow.appendChild(cell);
      });
      head.appendChild(headRow);
      const body = document.createElement('tbody');
      shown.forEach((values, rowIndex) => {
        const tr = document.createElement('tr');
        headers.forEach((_, columnIndex) => {
          const cell = document.createElement('td');
          cell.textContent = values[columnIndex] ?? '';
          cell.title = values[columnIndex] ?? '';
          tr.appendChild(cell);
        });
        tr.dataset.row = String(rowIndex + 1);
        body.appendChild(tr);
      });
      table.replaceChildren(head, body);
      summary.textContent = `${file.name} · ${Math.max(0, rows.length - 1).toLocaleString()} rows · ${headers.length} columns`;
    } catch (error) {
      table.innerHTML = '';
      const body = document.createElement('tbody');
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.className = 'preview-error';
      cell.textContent = error.message || 'Could not preview this CSV.';
      row.appendChild(cell);
      body.appendChild(row);
      table.appendChild(body);
    }
  };

  const renderSelection = () => {
    syncInputFiles();
    const files = [...selectedFiles.entries()];
    preview.hidden = files.length === 0;
    dropzone.hidden = files.length > 0;
    filenameLabel.textContent = files.length ? `${files.length} CSV file${files.length === 1 ? '' : 's'} selected` : 'No files selected';
    tabs.replaceChildren();
    if (!files.length) { activeKey = null; return; }
    if (!activeKey || !selectedFiles.has(activeKey)) activeKey = files[0][0];
    files.forEach(([key, file]) => {
      const tab = document.createElement('button');
      tab.type = 'button';
      tab.className = `file-tab${key === activeKey ? ' active' : ''}`;
      const name = document.createElement('span');
      name.textContent = file.name;
      const remove = document.createElement('i');
      remove.textContent = '×';
      remove.title = `Remove ${file.name}`;
      remove.addEventListener('click', (event) => {
        event.stopPropagation();
        selectedFiles.delete(key);
        if (activeKey === key) activeKey = null;
        renderSelection();
      });
      tab.append(name, remove);
      tab.addEventListener('click', () => { activeKey = key; renderSelection(); });
      tabs.appendChild(tab);
    });
    renderTable(selectedFiles.get(activeKey));
  };

  const addFiles = (files) => {
    [...files].filter((file) => file.name.toLowerCase().endsWith('.csv')).forEach((file) => {
      const key = fileKey(file);
      selectedFiles.set(key, file);
      activeKey = key;
    });
    renderSelection();
  };

  fileInput.addEventListener('change', () => addFiles(fileInput.files));
  addButton.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragging'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragging');
    addFiles(event.dataTransfer.files);
  });
}

const analysisForm = document.querySelector('[data-analysis-form]');
if (analysisForm) {
  analysisForm.addEventListener('submit', () => {
    const button = analysisForm.querySelector('.button-run');
    const state = analysisForm.querySelector('[data-running-state]');
    button.hidden = true;
    state.hidden = false;
  });
}

const tabs = document.querySelector('[data-tabs]');
if (tabs) {
  tabs.addEventListener('click', (event) => {
    const button = event.target.closest('[data-tab]');
    if (!button) return;
    tabs.querySelectorAll('[data-tab]').forEach((item) => item.classList.toggle('active', item === button));
    document.querySelectorAll('[data-panel]').forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === button.dataset.tab));
  });
}
