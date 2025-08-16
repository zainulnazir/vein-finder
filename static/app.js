(() => {
    const $ = (s) => document.querySelector(s);
    const $$ = (s) => Array.from(document.querySelectorAll(s));

    const showToast = (message, type = 'info', delay = 2200) => {
        const container = document.getElementById('toastContainer') || document.body;
        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
        toastEl.role = 'alert';
        toastEl.ariaLive = 'assertive';
        toastEl.ariaAtomic = 'true';
        toastEl.innerHTML = `<div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div>`;
        container.appendChild(toastEl);
        const t = new bootstrap.Toast(toastEl, { delay });
        t.show();
        toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
    };

    const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    const postJSON = async (url, payload) => {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload || {}),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    };

    const updateHUD = (settings = {}) => {
        if ('camera_exposure' in settings) setText('exposureInfo', `Exp: ${settings.camera_exposure}μs`);
        if ('camera_gain' in settings) setText('gainInfo', `Gain: ${settings.camera_gain}`);
        if ('zoom_level' in settings) setText('zoomInfo', `Zoom: ${Number(settings.zoom_level).toFixed(2)}x`);
        if ('live_processing_method' in settings || 'live_downscale' in settings) {
            const method = settings.live_processing_method || window.__liveMethod || 'adaptive';
            const scale = 'live_downscale' in settings ? Number(settings.live_downscale) : window.__liveScale || 0.5;
            setText('liveInfo', `Live: ${method.charAt(0).toUpperCase() + method.slice(1)} ${scale.toFixed(2)}x`);
        }
    if ('vein_tracking' in settings) setText('trackInfo', `Trk: ${settings.vein_tracking ? 'On' : 'Off'}`);
    };

    // Camera & image settings
    $('#applyCameraSettings')?.addEventListener('click', async () => {
        try {
            const payload = {
                detection_method: $('#detectionMethod').value,
                camera_exposure: Number($('#cameraExposure').value),
                camera_gain: Number($('#cameraGain').value),
                clahe_clip_limit: Number($('#claheClipLimit').value),
                clahe_tile_grid_size: Number($('#claheTileSize').value),
                stream_resolution: $('#streamResolution').value,
                median_ksize: Number($('#medianKsize')?.value || 3),
                adaptive_block_size: Number($('#adaptiveBlock')?.value || 31),
                adaptive_C: Number($('#adaptiveC')?.value || 2),
                morph_kernel: Number($('#morphKernel')?.value || 3),
            };
            const r = await postJSON('/update_settings', payload);
            if (r.success) {
                updateHUD(payload);
                showToast('Settings applied', 'success');
            }
        } catch (e) {
            showToast(`Apply failed: ${e.message}`, 'danger');
        }
    });

    // Live performance
    const liveScale = $('#liveScale');
    const streamFps = $('#streamFps');
    const jpegQ = $('#jpegQuality');
    const liveMethodRadios = $$('input[name="liveMethodRadios"]');
    const syncLiveLabels = () => {
        if (liveScale) setText('liveScaleValue', `${Number(liveScale.value).toFixed(2)}x`);
        if (streamFps) setText('streamFpsValue', `${streamFps.value}`);
        if (jpegQ) setText('jpegQualityValue', `${jpegQ.value}`);
    };
    [liveScale, streamFps, jpegQ].forEach((el) => el?.addEventListener('input', syncLiveLabels));
    liveMethodRadios.forEach((r) => r.addEventListener('change', () => {}));
    syncLiveLabels();

    $('#applyLiveSettings')?.addEventListener('click', async () => {
        try {
            const method = liveMethodRadios.find((r) => r.checked)?.value || 'adaptive';
            const payload = {
                live_downscale: Number(liveScale?.value || 0.5),
                live_processing_method: method,
                stream_target_fps: Number(streamFps?.value || 15),
                stream_jpeg_quality: Number(jpegQ?.value || 80),
            };
            const r = await postJSON('/update_settings', payload);
            if (r.success) {
                window.__liveMethod = method;
                window.__liveScale = payload.live_downscale;
                updateHUD(payload);
                showToast('Live settings applied', 'success');
                const img = $('#videoStream');
                const base = img.src.split('?')[0];
                img.src = `${base}?t=${Date.now()}`;
            }
        } catch (e) {
            showToast(`Apply failed: ${e.message}`, 'danger');
        }
    });

    // Capture
        $('#captureBtn')?.addEventListener('click', async () => {
        const btn = $('#captureBtn');
        const sp = $('#captureSpinner');
        try {
            btn.disabled = true;
            sp.classList.remove('d-none');
            const method = $('#detectionMethod')?.value || 'adaptive';
            const res = await postJSON('/capture', { detection_method: method });
            if (res.success) {
                const img = $('#lastCapturedImg');
                const ts = $('#captureTimestamp');
                const bust = `&_=${Date.now()}`;
                if (img && res.processed_image) img.src = `${res.processed_image}?${bust}`;
                const vp = $('#viewProcessedBtn');
                const vo = $('#viewOriginalBtn');
                const dl = $('#downloadBtn');
                if (vp && res.processed_image) vp.href = res.processed_image;
                if (vo && res.original_image) vo.href = res.original_image;
                if (dl && res.processed_image) dl.href = res.processed_image;
                if (ts) ts.textContent = res.timestamp || new Date().toLocaleString();
                // Refresh gallery after capture
                try { await loadGallery(); } catch {}
                showToast('Captured', 'success');
            } else {
                showToast('Capture failed', 'danger');
            }
        } catch (e) {
            showToast(`Capture error: ${e.message}`, 'danger');
        } finally {
            btn.disabled = false;
            sp.classList.add('d-none');
        }
    });

    // Zoom/Rotate/Reset
    $('#zoomInBtn')?.addEventListener('click', () => postJSON('/zoom', { action: 'in' }).catch(() => {}));
    $('#zoomOutBtn')?.addEventListener('click', () => postJSON('/zoom', { action: 'out' }).catch(() => {}));
    $('#rotateBtn')?.addEventListener('click', () => postJSON('/rotate', {}).catch(() => {}));
    $('#resetViewBtn')?.addEventListener('click', async () => {
        try {
            const r = await postJSON('/reset_view', {});
            if (r.success) {
                setText('zoomInfo', 'Zoom: 1.00x');
                showToast('View reset', 'info');
            }
        } catch (e) {
            showToast('Reset failed', 'danger');
        }
    });

    // Tracking
    $('#toggleTrackingBtn')?.addEventListener('click', async () => {
        try {
            const currently = (document.getElementById('trackInfo')?.textContent || '').includes('On');
            const r = await postJSON('/toggle_tracking', { enable: !currently });
            if (r.success) {
                setText('trackInfo', `Trk: ${r.vein_tracking ? 'On' : 'Off'}`);
                const btn = document.getElementById('toggleTrackingBtn');
                if (btn) btn.textContent = `Tracking: ${r.vein_tracking ? 'On' : 'Off'}`;
                showToast(`Tracking ${r.vein_tracking ? 'enabled' : 'disabled'}`, 'info');
            }
        } catch (e) {
            showToast('Tracking toggle failed', 'danger');
        }
    });

    // Click to set target for tracking
    $('#videoStream')?.addEventListener('click', async (ev) => {
        try {
            const rect = ev.target.getBoundingClientRect();
            const nx = (ev.clientX - rect.left) / rect.width;
            const ny = (ev.clientY - rect.top) / rect.height;
            await postJSON('/set_tracking_target', { x: nx, y: ny });
        } catch (e) {}
    });

    // LEDs
    $('#toggleLEDs')?.addEventListener('click', async () => {
        try { await postJSON('/toggle_leds', {}); showToast('LED toggle sent', 'info'); }
        catch { showToast('LED toggle failed', 'danger'); }
    });
    $('#applyLEDSettings')?.addEventListener('click', async () => {
        try {
            const payload = { led_brightness: Number($('#ledBrightness')?.value || 0), led_pattern: Number($('#ledPattern')?.value || 1) };
            await postJSON('/update_settings', payload);
            showToast('LED settings applied', 'success');
        } catch (e) { showToast('LED failed', 'danger'); }
    });

    // Range labels
    const wireRange = (inputSel, labelSel) => {
        const el = document.querySelector(inputSel);
        const lab = document.querySelector(labelSel);
        if (!el || !lab) return;
        const fn = () => (lab.textContent = el.value);
        el.addEventListener('input', fn);
        fn();
    };
    wireRange('#cameraExposure', '#exposureValue');
    wireRange('#cameraGain', '#gainValue');
    wireRange('#claheClipLimit', '#clipLimitValue');
    wireRange('#claheTileSize', '#tileSizeValue');
    wireRange('#medianKsize', '#medianValue');
    wireRange('#adaptiveBlock', '#adaptiveBlockValue');
    wireRange('#adaptiveC', '#adaptiveCValue');
    wireRange('#morphKernel', '#morphKernelValue');
    wireRange('#ledBrightness', '#brightnessValue');

    // Fullscreen
    $('#fullscreenBtn')?.addEventListener('click', () => {
        const el = document.documentElement;
        if (!document.fullscreenElement) el.requestFullscreen?.(); else document.exitFullscreen?.();
    });

    // Dark mode
    const applyDark = (on) => {
        document.body.classList.toggle('dark-mode', on);
        const icon = document.querySelector('#darkModeToggle i');
        if (icon) icon.className = on ? 'fas fa-sun' : 'fas fa-moon';
        localStorage.setItem('darkMode', on ? 'enabled' : 'disabled');
    };
    applyDark(localStorage.getItem('darkMode') === 'enabled');
    $('#darkModeToggle')?.addEventListener('click', () => applyDark(!document.body.classList.contains('dark-mode')));

    // Patient & Notes
    document.getElementById('patientForm')?.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        try {
            const payload = {
                name: document.getElementById('patientName')?.value || '',
                patient_id: document.getElementById('patientId')?.value || '',
                age: Number(document.getElementById('patientAge')?.value || 0) || null,
                procedure: document.getElementById('procedureType')?.value || '',
            };
            await postJSON('/save_patient', payload);
            showToast('Patient saved', 'success');
        } catch (e) {
            showToast('Save failed', 'danger');
        }
    });
    document.getElementById('saveNotesBtn')?.addEventListener('click', async () => {
        try {
            const notes = document.getElementById('procedureNotes')?.value || '';
            await postJSON('/save_notes', { notes });
            showToast('Notes saved', 'success');
        } catch (e) { showToast('Notes save failed', 'danger'); }
    });

    // Extra UI wiring: clear notes, hide last capture, suggest preset, shutdown
    document.getElementById('clearNotesBtn')?.addEventListener('click', (e) => {
        e.preventDefault();
        const ta = document.getElementById('procedureNotes');
        if (ta) ta.value = '';
    });
    document.getElementById('closeLastCapturedBtn')?.addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('lastCapturedContainer')?.classList.toggle('d-none');
    });
    document.getElementById('suggestPresetBtn')?.addEventListener('click', async () => {
        try {
            const r = await fetch('/suggest_preset');
            const j = await r.json();
            if (j.success && j.suggestion) {
                showToast(`Suggested: ${j.suggestion.name} – ${j.suggestion.reason || ''}`.trim(), 'info');
            } else {
                showToast('No suggestion available', 'secondary');
            }
        } catch {
            showToast('Suggest failed', 'danger');
        }
    });
    document.getElementById('confirmShutdown')?.addEventListener('click', async () => {
        try { const r = await postJSON('/shutdown', {}); if (r.success) showToast('Shutting down…', 'warning', 3000); } catch {}
    });

    // Profiles: load list, show description, load/save
    async function refreshProfiles() {
        try {
            const r = await fetch('/get_configurations');
            const j = await r.json();
            const sel = document.getElementById('profilesSelect');
            const desc = document.getElementById('profileDescription');
            if (!sel) return;
            sel.innerHTML = '<option value="">Select a profile…</option>';
            if (j.success && Array.isArray(j.presets)) {
                j.presets.forEach((p) => {
                    const opt = document.createElement('option');
                    opt.value = p.name;
                    opt.textContent = p.name;
                    opt.dataset.description = p.description || '';
                    sel.appendChild(opt);
                });
                sel.addEventListener('change', () => {
                    const o = sel.selectedOptions[0];
                    if (desc) desc.textContent = o?.dataset?.description || '';
                });
            }
        } catch (e) { /* ignore */ }
    }
    document.getElementById('loadProfileBtn')?.addEventListener('click', async () => {
        const sel = document.getElementById('profilesSelect');
        const name = sel?.value || '';
        if (!name) { showToast('Select a profile', 'secondary'); return; }
        try {
            const r = await postJSON('/load_configuration', { name });
            if (r.success) {
                // Reflect key settings in UI controls
                const s = r.settings || {};
                const setVal = (id, v) => { const el = document.getElementById(id); if (el && v !== undefined && v !== null) el.value = v; };
                setVal('detectionMethod', s.detection_method);
                setVal('cameraExposure', s.camera_exposure);
                setVal('cameraGain', s.camera_gain);
                setVal('claheClipLimit', s.clahe_clip_limit);
                setVal('claheTileSize', s.clahe_tile_grid_size);
                setVal('medianKsize', s.median_ksize);
                setVal('adaptiveBlock', s.adaptive_block_size);
                setVal('adaptiveC', s.adaptive_C);
                setVal('morphKernel', s.morph_kernel);
                setVal('streamResolution', s.stream_resolution);
                // Live method radios
                if (s.live_processing_method) {
                    const v = s.live_processing_method;
                    const radio = document.querySelector(`input[name="liveMethodRadios"][value="${v}"]`);
                    if (radio) radio.checked = true;
                }
                // Update range labels
                ['cameraExposure','cameraGain','claheClipLimit','claheTileSize','medianKsize','adaptiveBlock','adaptiveC','morphKernel'].forEach((k)=>{
                    document.getElementById(k)?.dispatchEvent(new Event('input'));
                });
                updateHUD({ camera_exposure: s.camera_exposure, camera_gain: s.camera_gain, zoom_level: s.zoom_level, live_processing_method: s.live_processing_method, live_downscale: s.live_downscale });
                // Refresh stream to pick up possible resolution/encoding changes
                const img = document.getElementById('videoStream');
                if (img && img.src) { const base = img.src.split('?')[0]; img.src = `${base}?t=${Date.now()}`; }
                showToast(`Loaded profile: ${name}`, 'success');
            } else {
                showToast(r.message || 'Load failed', 'danger');
            }
        } catch (e) { showToast('Load failed', 'danger'); }
    });
    document.getElementById('saveProfileBtn')?.addEventListener('click', async () => {
        const name = (document.getElementById('saveProfileName')?.value || '').trim();
        const description = (document.getElementById('saveProfileDesc')?.value || '').trim();
        if (!name) { showToast('Enter a profile name', 'secondary'); return; }
        try {
            // First push current UI settings to backend so the saved profile matches the sliders
            const payload = {
                detection_method: document.getElementById('detectionMethod')?.value,
                camera_exposure: Number(document.getElementById('cameraExposure')?.value || 0),
                camera_gain: Number(document.getElementById('cameraGain')?.value || 0),
                clahe_clip_limit: Number(document.getElementById('claheClipLimit')?.value || 0),
                clahe_tile_grid_size: Number(document.getElementById('claheTileSize')?.value || 0),
                median_ksize: Number(document.getElementById('medianKsize')?.value || 0),
                adaptive_block_size: Number(document.getElementById('adaptiveBlock')?.value || 0),
                adaptive_C: Number(document.getElementById('adaptiveC')?.value || 0),
                morph_kernel: Number(document.getElementById('morphKernel')?.value || 0),
                stream_resolution: document.getElementById('streamResolution')?.value
            };
            await postJSON('/update_settings', payload);
            const r = await postJSON('/save_configuration', { name, description });
            if (r.success) { showToast('Profile saved', 'success'); refreshProfiles(); }
            else { showToast(r.message || 'Save failed', 'danger'); }
        } catch (e) { showToast('Save failed', 'danger'); }
    });
    refreshProfiles();

    // Image Gallery state & helpers
    const gallery = {
        items: [],
        filtered: [],
        page: 1,
        pageSize: (() => { const sel = document.getElementById('galleryPageSize'); return sel ? parseInt(sel.value || '12', 10) : 12; })(),
        query: ''
    };

    const setPaginationInfo = () => {
        const info = document.getElementById('paginationInfo');
        if (!info) return;
        const total = Math.max(1, Math.ceil(gallery.filtered.length / gallery.pageSize));
        if (gallery.page > total) gallery.page = total;
        info.textContent = `Page ${gallery.page}/${total}`;
        const prev = document.getElementById('paginationPrev');
        const next = document.getElementById('paginationNext');
        if (prev) prev.disabled = gallery.page <= 1;
        if (next) next.disabled = gallery.page >= total;
    };

    const renderGallery = () => {
        const grid = document.getElementById('imageGallery');
        if (!grid) return;
        grid.innerHTML = '';
        if (!gallery.filtered.length) {
            grid.innerHTML = '<div class="col-12 text-center text-muted small py-4">No images</div>';
            setPaginationInfo();
            return;
        }
        const start = (gallery.page - 1) * gallery.pageSize;
        const end = Math.min(start + gallery.pageSize, gallery.filtered.length);
        const pageItems = gallery.filtered.slice(start, end);
        const frag = document.createDocumentFragment();
        pageItems.forEach((it) => {
            const col = document.createElement('div');
            col.className = 'col-6 col-md-4 col-lg-3';
            col.innerHTML = `
                <div class="card h-100">
                  <img src="${it.processed_image}" class="card-img-top" alt="${it.timestamp}">
                  <div class="card-body p-2 small d-flex flex-column gap-1">
                    <div class="d-flex justify-content-between align-items-center">
                      <span class="text-muted">${it.timestamp}</span>
                      <div class="btn-group btn-group-sm">
                        <a class="btn btn-outline-secondary" target="_blank" href="${it.original_image || it.processed_image}">Orig</a>
                        <a class="btn btn-outline-secondary" target="_blank" href="${it.processed_image}">Proc</a>
                        <a class="btn btn-success" target="_blank" download href="${it.processed_image}"><i class="fas fa-download"></i></a>
                        <button class="btn btn-outline-danger btn-del" data-url="${it.processed_image}"><i class="fas fa-trash"></i></button>
                      </div>
                    </div>
                  </div>
                </div>`;
            frag.appendChild(col);
        });
        grid.appendChild(frag);
        // Wire delete buttons
        grid.querySelectorAll('.btn-del').forEach((btn) => {
            btn.addEventListener('click', async (e) => {
                const url = e.currentTarget.getAttribute('data-url');
                if (!url) return;
                if (!confirm('Delete this image and associated files?')) return;
                try {
                    const r = await postJSON('/delete_image', { image_path: url });
                    if (r.success) {
                        showToast('Deleted', 'warning');
                        await loadGallery();
                    } else {
                        showToast('Delete failed', 'danger');
                    }
                } catch { showToast('Delete failed', 'danger'); }
            });
        });
        setPaginationInfo();
    };

    const applySearch = async () => {
        const q = (document.getElementById('gallerySearch')?.value || '').trim();
        gallery.query = q;
        gallery.page = 1;
        if (!q) {
            gallery.filtered = gallery.items.slice();
            renderGallery();
            return;
        }
        try {
            const res = await postJSON('/search_images', { query: q });
            if (res.success) {
                gallery.filtered = res.images || [];
                renderGallery();
            }
        } catch { /* keep previous */ }
    };

    async function loadGallery() {
        const grid = document.getElementById('imageGallery');
        if (grid) grid.innerHTML = '<div class="col-12 text-center py-4"><div class="spinner-border spinner-border-sm text-primary" role="status"></div><p class="small text-muted mb-0 mt-2">Loading images...</p></div>';
        try {
            const res = await fetch('/images');
            if (!res.ok) throw new Error('Failed to load images');
            const j = await res.json();
            if (j.success) {
                gallery.items = j.images || [];
                gallery.filtered = gallery.items.slice();
                gallery.page = 1;
                renderGallery();
            } else {
                grid.innerHTML = '<div class="col-12 text-center text-danger small py-4">Failed to load images</div>';
            }
        } catch (e) {
            if (grid) grid.innerHTML = `<div class="col-12 text-center text-danger small py-4">${e.message}</div>`;
        }
    }

    // Wire gallery controls
    document.getElementById('gallerySearchBtn')?.addEventListener('click', (e) => { e.preventDefault(); applySearch(); });
    document.getElementById('gallerySearch')?.addEventListener('keypress', (e) => { if (e.key === 'Enter') { e.preventDefault(); applySearch(); }});
    document.getElementById('galleryPageSize')?.addEventListener('change', (e) => {
        gallery.pageSize = parseInt(e.target.value || '12', 10);
        gallery.page = 1;
        renderGallery();
    });
    document.getElementById('paginationPrev')?.addEventListener('click', (e) => { e.preventDefault(); if (gallery.page > 1) { gallery.page -= 1; renderGallery(); } });
    document.getElementById('paginationNext')?.addEventListener('click', (e) => {
        e.preventDefault();
        const total = Math.max(1, Math.ceil(gallery.filtered.length / gallery.pageSize));
        if (gallery.page < total) { gallery.page += 1; renderGallery(); }
    });
    document.getElementById('refreshGalleryBtn')?.addEventListener('click', async (e) => { e.preventDefault(); (document.getElementById('gallerySearch').value = ''); await loadGallery(); });
    document.getElementById('clearGalleryBtn')?.addEventListener('click', async (e) => {
        e.preventDefault();
        if (!confirm('Clear all images?')) return;
        try { const r = await postJSON('/clear_gallery', {}); if (r.success) { showToast('Gallery cleared', 'warning'); await loadGallery(); } else { showToast('Clear failed', 'danger'); } } catch { showToast('Clear failed', 'danger'); }
    });

    // Initial load
    loadGallery();
})();