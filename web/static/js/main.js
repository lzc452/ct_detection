(function() {
    'use strict';

    // ── 黑客帝国代码雨背景 ──
    (function initMatrix() {
        const canvas = document.getElementById('matrixBg');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        let w, h, cols;
        const fontSize = 14;
        let drops = [];
        const chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzαβγδεζηθικλμνξοπρστυφχψωあいうえおアイウエオｱｲｳｴｵﾊﾋﾌﾍﾎﾀﾁﾂﾃﾄ';

        function resize() {
            w = canvas.width = window.innerWidth;
            h = canvas.height = window.innerHeight;
            cols = Math.floor(w / fontSize);
            drops = [];
            for (let i = 0; i < cols; i++) {
                drops[i] = Math.random() * -100;
            }
        }

        resize();
        window.addEventListener('resize', resize);

        function draw() {
            ctx.fillStyle = 'rgba(250, 249, 246, 0.08)';
            ctx.fillRect(0, 0, w, h);

            ctx.shadowBlur = 4;
            ctx.shadowColor = '#00FF00';
            ctx.fillStyle = '#00FF00';
            ctx.font = fontSize + 'px monospace';

            for (let i = 0; i < drops.length; i++) {
                const text = chars.charAt(Math.floor(Math.random() * chars.length));
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);

                if (drops[i] * fontSize > h && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
            ctx.shadowBlur = 0;
        }

        let animId;
        function loop() {
            draw();
            animId = requestAnimationFrame(loop);
        }
        loop();

        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                cancelAnimationFrame(animId);
            } else {
                loop();
            }
        });
    })();

    // 全局状态
    let allResults = [];      // 所有检测结果
    let currentMode = 'none'; // 'image', 'video', 'batch'
    let isProcessing = false;
    let abortBatch = false;   // 批量检测中断标志

    // 类别名称
    const ConfigNames = ['damage', 'dark', 'light'];
    const ConfigCHNames = ['极片损伤', '暗斑', '亮斑'];

    // DOM 元素
    const els = {
        btnPic: document.getElementById('btnPic'),
        btnVideo: document.getElementById('btnVideo'),
        btnFolder: document.getElementById('btnFolder'),
        btnSmart: document.getElementById('btnSmart'),
        fileInput: document.getElementById('fileInput'),
        videoInput: document.getElementById('videoInput'),
        folderInput: document.getElementById('folderInput'),
        smartInput: document.getElementById('smartInput'),
        uploadZone: document.getElementById('uploadZone'),
        uploadContent: document.getElementById('uploadContent'),
        resultImage: document.getElementById('resultImage'),
        resultVideo: document.getElementById('resultVideo'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingText: document.getElementById('loadingText'),
        progressOverlay: document.getElementById('progressOverlay'),
        progressFill: document.getElementById('progressFill'),
        progressText: document.getElementById('progressText'),
        progressFilename: document.getElementById('progressFilename'),
        progressCount: document.getElementById('progressCount'),
        progressStopBtn: document.getElementById('progressStopBtn'),
        scanLineOverlay: document.getElementById('scanLineOverlay'),
        resultTableBody: document.getElementById('resultTableBody'),
        confSlider: document.getElementById('confSlider'),
        confDisplay: document.getElementById('confDisplay'),
        confWarn: document.getElementById('confWarn'),
        timeValue: document.getElementById('timeValue'),
        targetCount: document.getElementById('targetCount'),
        confValue: document.getElementById('confValue'),
        targetSelect: document.getElementById('targetSelect'),
        xmin: document.getElementById('xmin'),
        ymin: document.getElementById('ymin'),
        xmax: document.getElementById('xmax'),
        ymax: document.getElementById('ymax'),
        btnDownload: document.getElementById('btnDownload'),
        btnExport: document.getElementById('btnExport'),
        btnClear: document.getElementById('btnClear'),
        recordCount: document.getElementById('recordCount'),
        fullscreenBtn: document.getElementById('fullscreenBtn'),
        toggleOrigBtn: document.getElementById('toggleOrigBtn'),
        fullscreenOverlay: document.getElementById('fullscreenOverlay'),
        fullscreenImage: document.getElementById('fullscreenImage'),
        fullscreenClose: document.getElementById('fullscreenClose'),
        filterClass: document.getElementById('filterClass'),
        btnExpand: document.getElementById('btnExpand'),
        fullscreenTableOverlay: document.getElementById('fullscreenTableOverlay'),
        fullscreenTableBodyInner: document.getElementById('fullscreenTableBodyInner'),
        fullscreenTableClose: document.getElementById('fullscreenTableClose'),
        modelSelect: document.getElementById('modelSelect'),
        modelStatus: document.getElementById('modelStatus'),
        bootOverlay: document.getElementById('bootOverlay'),
        bootStage: document.getElementById('bootStage'),
        bootFill: document.getElementById('bootFill'),
        bootPct: document.getElementById('bootPct'),
        pathModal: document.getElementById('pathModal'),
        pathModalClose: document.getElementById('pathModalClose'),
        pathInput: document.getElementById('pathInput'),
        btnBrowsePath: document.getElementById('btnBrowsePath'),
        btnScanPath: document.getElementById('btnScanPath'),
        groupPanel: document.getElementById('groupPanel'),
        groupSummary: document.getElementById('groupSummary'),
        groupOps: document.getElementById('groupOps'),
        groupList: document.getElementById('groupList'),
        groupSelectAll: document.getElementById('groupSelectAll'),
        groupSelectNone: document.getElementById('groupSelectNone'),
        btnStartDetect: document.getElementById('btnStartDetect'),
        detectFloat: document.getElementById('detectFloat'),
        detectFloatFile: document.getElementById('detectFloatFile'),
        detectFloatFill: document.getElementById('detectFloatFill'),
        detectFloatStats: document.getElementById('detectFloatStats'),
        detectStopBtn: document.getElementById('detectStopBtn'),
        backLivePill: document.getElementById('backLivePill'),
        btnHistory: document.getElementById('btnHistory'),
        historyModal: document.getElementById('historyModal'),
        historyClose: document.getElementById('historyClose'),
        historyList: document.getElementById('historyList'),
        clearModal: document.getElementById('clearModal'),
        clearModalMsg: document.getElementById('clearModalMsg'),
        clearCancel: document.getElementById('clearCancel'),
        clearConfirm: document.getElementById('clearConfirm'),
    };

    // 检测执行期间禁用所有动作按钮（防重入/防抖），结束后恢复
    const actionBtns = () => [els.btnPic, els.btnVideo, els.btnFolder, els.btnSmart,
                              els.btnDownload, els.btnExport, els.btnClear, els.progressStopBtn];
    function setProcessing(v) {
        isProcessing = v;
        actionBtns().forEach(b => {
            if (!b) return;
            if (b === els.progressStopBtn) return;   // 停止按钮另有显隐逻辑
            if (v) {
                b.dataset.prevDisabled = b.disabled ? '1' : '';
                b.disabled = true;
                b.style.opacity = '0.5';
            } else {
                b.disabled = b.dataset.prevDisabled === '1';
                b.style.opacity = b.disabled ? '0.5' : '';
            }
        });
    }

    // 参数同步（#8：阈值直传模型，0.05~1.00 都可生效；低于 0.25 显著增加误报）
    function syncConfHint() {
        const v = els.confSlider.value / 100;
        els.confDisplay.textContent = v.toFixed(2);
        if (els.confWarn) {
            els.confWarn.style.display = v < 0.25 ? 'block' : 'none';
        }
    }
    els.confSlider.addEventListener('input', syncConfHint);
    syncConfHint();

    // 显示/隐藏 加载
    function showLoading(text) {
        els.loadingText.textContent = text || 'Processing...';
        els.loadingOverlay.style.display = 'flex';
    }
    function hideLoading() {
        els.loadingOverlay.style.display = 'none';
    }

    // 显示进度条弹窗
    function showProgress(total) {
        els.progressOverlay.style.display = 'flex';
        els.progressFill.style.width = '0%';
        els.progressText.textContent = '0%';
        els.progressFilename.textContent = '准备中...';
        if (total) {
            els.progressCount.textContent = '0 / ' + total;
        } else {
            els.progressCount.textContent = '';
        }
        abortBatch = false;
    }
    function updateProgress(pct, text, filename, current, total) {
        els.progressFill.style.width = pct + '%';
        els.progressText.textContent = text || (pct + '%');
        if (filename) {
            els.progressFilename.textContent = filename;
        }
        if (current !== undefined && total !== undefined) {
            els.progressCount.textContent = current + ' / ' + total;
        }
    }
    function hideProgress() {
        els.progressOverlay.style.display = 'none';
        els.progressFilename.textContent = '';
        els.progressCount.textContent = '';
    }

    // 停止按钮事件
    els.progressStopBtn.addEventListener('click', () => {
        abortBatch = true;
        hideProgress();
    });

    // 检测框脉冲呼吸
    function startDetectBreathing() {
        document.getElementById('displayCard').classList.add('detect-breathing');
    }
    function stopDetectBreathing() {
        document.getElementById('displayCard').classList.remove('detect-breathing');
    }

    // 表格行滑入动画
    function animateNewRows() {
        const rows = els.resultTableBody.querySelectorAll('tr:not(.empty-row)');
        rows.forEach((row, idx) => {
            if (!row.classList.contains('slide-in-row')) {
                row.classList.add('slide-in-row');
                row.style.animationDelay = (idx * 0.05) + 's';
            }
        });
    }

    // 扫描线特效
    function showScanLine() {
        els.scanLineOverlay.style.display = 'block';
    }
    function hideScanLine() {
        els.scanLineOverlay.style.display = 'none';
    }

    // 数字滚动效果
    function animateNumber(el, targetValue, duration = 600) {
        if (!el || targetValue === '--') return;
        const start = 0;
        const startTime = performance.now();
        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (parseFloat(targetValue) - start) * ease);
            el.textContent = current;
            if (progress < 1) requestAnimationFrame(step);
            else {
                el.textContent = targetValue;
                el.classList.add('number-flash');
                setTimeout(() => el.classList.remove('number-flash'), 600);
            }
        }
        requestAnimationFrame(step);
    }

    // Toast 通知
    function showToast(type, title, message, duration = 3000) {
        const container = document.getElementById('toastContainer');
        const icons = { success: '✓', error: '✕', info: '◉' };
        const toast = document.createElement('div');
        toast.className = 'toast-item';
        toast.innerHTML = `
            <div class="toast-icon ${type}">${icons[type]}</div>
            <div class="toast-body">
                <div class="toast-title">${title}</div>
                <div class="toast-msg">${message}</div>
            </div>
        `;
        container.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('toast-out');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // 扫描完成闪光
    function flashScanComplete() {
        els.uploadZone.classList.add('scan-flash');
        setTimeout(() => els.uploadZone.classList.remove('scan-flash'), 500);
    }

    // 显示图片
    function showImage(url) {
        if (!url) return;
        els.resultImage.src = url;
        els.resultImage.style.display = 'block';
        els.resultVideo.style.display = 'none';
        els.uploadContent.style.display = 'none';
        els.fullscreenBtn.style.display = 'flex';
        currentMode = 'image';
    }
    // 显示视频
    function showVideo(url) {
        if (!url) return;
        els.resultVideo.src = url;
        els.resultVideo.style.display = 'block';
        els.resultImage.style.display = 'none';
        els.uploadContent.style.display = 'none';
        els.fullscreenBtn.style.display = 'none';
        currentMode = 'video';
    }
    // 清空显示
    function clearDisplay() {
        els.resultImage.style.display = 'none';
        els.resultImage.src = '';
        els.resultVideo.style.display = 'none';
        els.resultVideo.src = '';
        els.uploadContent.style.display = 'flex';
        els.fullscreenBtn.style.display = 'none';
        els.toggleOrigBtn.style.display = 'none';
        currentMode = 'none';
    }

    // ── 原图/标注图切换（标注框会遮挡细节，问题图可同时看两版）──
    const viewPair = { ann: '', orig: '', showing: 'ann' };
    function showResultImage(annUrl, origUrl) {
        viewPair.ann = annUrl || '';
        viewPair.orig = origUrl || '';
        viewPair.showing = 'ann';
        showImage(viewPair.ann || viewPair.orig);
        const canToggle = !!(viewPair.ann && viewPair.orig);
        els.toggleOrigBtn.style.display = canToggle ? 'flex' : 'none';
        if (canToggle) els.toggleOrigBtn.querySelector('span').textContent = '原图';
    }
    els.toggleOrigBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!viewPair.ann || !viewPair.orig) return;
        viewPair.showing = viewPair.showing === 'ann' ? 'orig' : 'ann';
        els.resultImage.src = viewPair.showing === 'ann' ? viewPair.ann : viewPair.orig;
        els.toggleOrigBtn.querySelector('span').textContent = viewPair.showing === 'ann' ? '原图' : '标注图';
    });

    // 更新结果标签
    function updateResultLabels(time, count, conf, locations) {
        els.timeValue.textContent = time !== null && time !== undefined ? time : '--';
        els.targetCount.textContent = count !== undefined ? count : '--';
        els.confValue.textContent = conf || '--';
        if (locations && locations.length > 0) {
            els.xmin.value = locations[0][0];
            els.ymin.value = locations[0][1];
            els.xmax.value = locations[0][2];
            els.ymax.value = locations[0][3];
        } else {
            els.xmin.value = '--'; els.ymin.value = '--';
            els.xmax.value = '--'; els.ymax.value = '--';
        }
    }

    // 更新目标选择下拉框
    function updateTargetSelect(classes, confs, locations) {
        els.targetSelect.innerHTML = '<option value="-1">全部类别</option>';
        if (!classes || classes.length === 0) return;
        classes.forEach((clsId, idx) => {
            const name = ConfigCHNames[clsId] || ConfigNames[clsId] || ('class_' + clsId);
            const option = document.createElement('option');
            option.value = idx;
            option.textContent = name + '_' + idx;
            els.targetSelect.appendChild(option);
        });
        els.targetSelect.dataset.classes = JSON.stringify(classes);
        els.targetSelect.dataset.confs = JSON.stringify(confs);
        els.targetSelect.dataset.locations = JSON.stringify(locations);
    }

    els.targetSelect.addEventListener('change', () => {
        const idx = parseInt(els.targetSelect.value);
        const confs = JSON.parse(els.targetSelect.dataset.confs || '[]');
        const locations = JSON.parse(els.targetSelect.dataset.locations || '[]');
        if (idx === -1) {
            if (confs.length > 0) {
                els.confValue.textContent = confs[0];
                els.xmin.value = locations[0][0];
                els.ymin.value = locations[0][1];
                els.xmax.value = locations[0][2];
                els.ymax.value = locations[0][3];
            }
        } else {
            if (confs[idx] !== undefined) {
                els.confValue.textContent = confs[idx];
                els.xmin.value = locations[idx][0];
                els.ymin.value = locations[idx][1];
                els.xmax.value = locations[idx][2];
                els.ymax.value = locations[idx][3];
            }
        }
    });

    // 高亮行
    function highlightRow(row) {
        els.resultTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
    }

    // 清除空行提示
    function removeEmptyRow() {
        const emptyRow = els.resultTableBody.querySelector('.empty-row');
        if (emptyRow) emptyRow.remove();
    }

    // 获取当前表格中已有的图片数量（去重）
    function getCurrentImageCount() {
        const seen = new Set();
        els.resultTableBody.querySelectorAll('tr').forEach(r => {
            if (r.dataset.filename) seen.add(r.dataset.filename);
        });
        return seen.size;
    }

    // 获取当前表格中已有的目标行数
    function getCurrentRowCount() {
        let count = 0;
        els.resultTableBody.querySelectorAll('tr').forEach(r => {
            const firstTd = r.querySelector('td:first-child');
            if (firstTd && firstTd.textContent.match(/^\d+/)) {
                count++;
            }
        });
        return count;
    }

    // 添加表格行（一图一序号，多缺陷分叉）
    function addTableRows(detection, imageIndex) {
        removeEmptyRow();
        const tbody = els.resultTableBody;
        const locations = detection.locations || [];
        const classes = detection.classes || [];
        const confs = detection.confs || [];
        const filename = detection.filename || '';
        const imageUrl = detection.image_url || '';
        const origUrl = detection.orig_url || '';
        const time = detection.time || '--';
        const status = detection.status || '';
        const hasTargets = locations.length > 0;

        // 检测失败（#3）：必须与"未检出目标"完全区分，否则坏图会被当成良品流下去
        if (status === 'failed') {
            const row = document.createElement('tr');
            row.className = 'row-failed';
            const reason = detection.error ? `（${escapeHtml(detection.error)}）` : '';
            row.innerHTML = `
                <td colspan="5">⚠ ${imageIndex} - 检测失败：${escapeHtml(filename)}${reason}</td>
                <td class="cell-del"><button class="row-del" title="删除这条失败记录">✕</button></td>
            `;
            row.dataset.status = 'failed';
            row.dataset.batchId = detection.batch_id || '';
            row.dataset.time = time;
            row.dataset.targetCount = 0;
            row.dataset.filename = filename;
            row.dataset.imageIndex = imageIndex;
            const delBtn = row.querySelector('.row-del');
            if (delBtn) {
                delBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    deleteImageRecord(row.dataset.batchId, filename);
                });
            }
            row.addEventListener('click', () => {
                highlightRow(row);
                els.toggleOrigBtn.style.display = 'none';
                updateResultLabels(time, 0, null, []);
                updateTargetSelect([], [], []);
            });
            tbody.appendChild(row);
            tbody.scrollTop = tbody.scrollHeight;
            return;
        }

        if (!hasTargets) {
            const row = document.createElement('tr');
            row.className = 'no-target';
            row.innerHTML = `<td colspan="6"><span style="color:#8A8580;">${imageIndex} - 未检出目标：${escapeHtml(filename)}</span></td>`;
            row.dataset.imageUrl = imageUrl;
            row.dataset.origUrl = origUrl;
            row.dataset.batchId = detection.batch_id || '';
            row.dataset.time = time;
            row.dataset.targetCount = 0;
            row.dataset.filename = filename;
            row.dataset.imageIndex = imageIndex;
            row.addEventListener('click', () => {
                highlightRow(row);
                if (row.dataset.imageUrl) showImage(row.dataset.imageUrl);
                els.toggleOrigBtn.style.display = 'none';
                updateResultLabels(row.dataset.time, 0, null, []);
                updateTargetSelect([], [], []);
            });
            tbody.appendChild(row);
            return;
        }

        locations.forEach((loc, idx) => {
            const row = document.createElement('tr');
            const displayIndex = idx === 0 ? `${imageIndex}` : `${imageIndex}-${idx}`;
            const clsId = classes[idx];
            const conf = confs[idx];
            const clsName = ConfigCHNames[clsId] || '未知';

            if (idx > 0) {
                row.classList.add('sub-defect');
            }

            row.innerHTML = `
                <td>${displayIndex}</td>
                <td>${escapeHtml(filename)}</td>
                <td>${escapeHtml(clsName)}</td>
                <td>${escapeHtml(conf)}</td>
                <td>[${loc.join(', ')}]</td>
                <td class="cell-del">${idx === 0 ? '<button class="row-del" title="删除这张图的所有记录（含备份文件）">✕</button>' : ''}</td>
            `;
            row.dataset.xmin = loc[0];
            row.dataset.ymin = loc[1];
            row.dataset.xmax = loc[2];
            row.dataset.ymax = loc[3];
            row.dataset.conf = conf;
            row.dataset.time = time;
            row.dataset.targetCount = locations.length;
            row.dataset.clsName = clsName;
            row.dataset.imageUrl = imageUrl;
            row.dataset.origUrl = origUrl;
            row.dataset.batchId = detection.batch_id || '';
            row.dataset.filename = filename;
            row.dataset.imageIndex = imageIndex;
            row.dataset.defectIndex = idx;

            if (idx === 0) {
                const delBtn = row.querySelector('.row-del');
                if (delBtn) {
                    delBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        deleteImageRecord(row.dataset.batchId, filename);
                    });
                }
            }

            row.addEventListener('click', () => {
                highlightRow(row);
                if (row.dataset.imageUrl) showResultImage(row.dataset.imageUrl, row.dataset.origUrl);

                els.timeValue.textContent = row.dataset.time || '--';
                els.targetCount.textContent = row.dataset.targetCount || '--';
                els.confValue.textContent = row.dataset.conf || '--';
                els.xmin.value = row.dataset.xmin || '--';
                els.ymin.value = row.dataset.ymin || '--';
                els.xmax.value = row.dataset.xmax || '--';
                els.ymax.value = row.dataset.ymax || '--';

                updateTargetSelect(classes, confs, locations);
                els.targetSelect.value = idx;
            });

            tbody.appendChild(row);
        });

        tbody.scrollTop = tbody.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function updateRecordCount() {
        const count = getCurrentRowCount();
        els.recordCount.textContent = '共 ' + count + ' 条记录';
    }

    // ── 筛选相关 ──
    function updateFilterOptions() {
        const rows = els.resultTableBody.querySelectorAll('tr[data-cls-name]');
        const classSet = new Set();
        rows.forEach(r => {
            const cls = r.dataset.clsName;
            if (cls) classSet.add(cls);
        });
        const currentVal = els.filterClass.value;
        const sorted = Array.from(classSet).sort();
        let html = '<option value="all">全部</option>';
        sorted.forEach(c => {
            html += `<option value="${c}">${c}</option>`;
        });
        els.filterClass.innerHTML = html;
        if (sorted.includes(currentVal) || currentVal === 'all') {
            els.filterClass.value = currentVal;
        } else {
            els.filterClass.value = 'all';
        }
    }

    function applyFilter() {
        const classFilter = els.filterClass.value;
        const hasConfOnly = true;   // 2026-10-21：表格默认只显示问题图，勾选项已移除、固定生效
        const rows = els.resultTableBody.querySelectorAll('tr');

        let visibleCount = 0;
        rows.forEach(row => {
            if (row.classList.contains('empty-row')) return;

            const targetCount = parseInt(row.dataset.targetCount || '0');
            const clsName = row.dataset.clsName || '';

            // 失败行永远显示（无类别筛选时）：它是需要人工处理的异常，
            // 不能被"只显示问题图"规则连坐隐藏（#3）
            if (row.dataset.status === 'failed') {
                const showFailed = classFilter === 'all';
                row.style.display = showFailed ? '' : 'none';
                if (showFailed) visibleCount++;
                return;
            }

            let show = true;
            if (hasConfOnly && targetCount === 0) {
                show = false;
            }
            if (classFilter !== 'all' && clsName !== classFilter) {
                show = false;
            }

            row.style.display = show ? '' : 'none';
            if (show) visibleCount++;
        });

        els.recordCount.textContent = '共 ' + visibleCount + ' 条记录';
    }

    els.filterClass.addEventListener('change', applyFilter);

    // ════════════════════════════════════════
    // 单张图片检测（文件上传）
    // ════════════════════════════════════════
    els.btnPic.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', async () => {
        const file = els.fileInput.files[0];
        if (!file || isProcessing) return;
        await detectImageUpload(file);
        els.fileInput.value = '';
    });

    async function detectImageUpload(file) {
        if (isProcessing) return;
        setProcessing(true);
        showScanLine();

        const form = new FormData();
        form.append('file', file);
        form.append('conf', (els.confSlider.value / 100).toFixed(2));
        const bid = 'B' + Date.now().toString(36);
        form.append('batch_id', bid);
        form.append('file_paths', JSON.stringify([file.webkitRelativePath || file.name]));
        switchToLive(bid);

        showLoading('Detecting...');

        try {
            const res = await fetch('/api/detect/image', { method: 'POST', body: form });
            const data = await res.json();
            hideLoading();
            hideScanLine();
            flashScanComplete();

            if (!data.success) {
                showToast('error', '检测失败', data.message || '图片检测失败');
                clearDisplay();
                setProcessing(false);
                return;
            }

            showResultImage(data.image_url, data.orig_url);
            updateResultLabels(data.time, data.target_nums, data.confs[0], data.locations);
            updateTargetSelect(data.classes, data.confs, data.locations);

            const imageIndex = getCurrentImageCount() + 1;
            const detection = {
                filename: data.filename,
                status: data.status || '',
                locations: data.locations,
                classes: data.classes,
                confs: data.confs,
                image_url: data.image_url,
                orig_url: data.orig_url,
                time: data.time
            };
            allResults.push(detection);
            addTableRows(detection, imageIndex);
            animateNewRows();
            updateRecordCount();
            updateFilterOptions();
            applyFilter();

            if (data.target_nums > 0) {
                startDetectBreathing();
                showToast('success', '检测完成', `发现 ${data.target_nums} 个缺陷`);
            } else {
                showToast('info', '检测完成', '未检测到缺陷');
            }
            setTimeout(stopDetectBreathing, 3000);

        } catch (err) {
            hideLoading();
            hideScanLine();
            showToast('error', '网络错误', '请求失败，请检查后端服务');
            console.error(err);
            clearDisplay();
            setProcessing(false);
        } finally {
            setProcessing(false);
        }
    }

    // ════════════════════════════════════════
    // 视频检测（文件上传）
    // ════════════════════════════════════════
    els.btnVideo.addEventListener('click', () => els.videoInput.click());
    els.videoInput.addEventListener('change', async () => {
        const file = els.videoInput.files[0];
        if (!file || isProcessing) return;
        await detectVideoUpload(file);
        els.videoInput.value = '';
    });

    async function detectVideoUpload(file) {
        if (isProcessing) return;
        setProcessing(true);

        const form = new FormData();
        form.append('file', file);
        form.append('conf', (els.confSlider.value / 100).toFixed(2));

        showLoading('Uploading video...');
        showProgress();

        try {
            const res = await fetch('/api/detect/video', { method: 'POST', body: form });
            const data = await res.json();
            hideLoading();

            if (!data.success) {
                showToast('error', '上传失败', data.message || '视频上传失败');
                hideProgress();
                clearDisplay();
                setProcessing(false);
                return;
            }

            const taskId = data.task_id;
            const poll = setInterval(async () => {
                const statusRes = await fetch(`/api/task/${taskId}`);
                const status = await statusRes.json();
                updateProgress(status.progress, status.progress + '%');

                if (status.status === 'done') {
                    clearInterval(poll);
                    hideProgress();
                    showVideo(status.result_url);
                    updateResultLabels('--', '--', '--', []);
                    updateTargetSelect([], [], []);

                    const detection = {
                        filename: file.name, locations: [], classes: [], confs: [],
                        video_url: status.result_url, time: '--'
                    };
                    allResults.push(detection);

                    removeEmptyRow();
                    const row = document.createElement('tr');
                    row.innerHTML = `<td colspan="6" style="color:#A0834B; font-weight:500; text-align:center; padding:10px;">Video processed: ${escapeHtml(file.name)}</td>`;
                    row.dataset.targetCount = '0';
                    els.resultTableBody.appendChild(row);
                    els.resultTableBody.scrollTop = els.resultTableBody.scrollHeight;
                    updateRecordCount();
                    updateFilterOptions();
                    applyFilter();
                    setProcessing(false);

                } else if (status.status === 'error') {
                    clearInterval(poll);
                    hideProgress();
                    hideScanLine();
                    showToast('error', '视频处理失败', status.message || '处理失败');
                    clearDisplay();
                    setProcessing(false);
                }
            }, 700);

        } catch (err) {
            hideLoading();
            hideProgress();
            hideScanLine();
            showToast('error', '网络错误', '视频上传失败');
            console.error(err);
            clearDisplay();
            setProcessing(false);
        }
    }

    // ════════════════════════════════════════
    // 批量检测（文件夹选择，一次性发送所有文件，后端 batch 推理）
    // ════════════════════════════════════════
    els.btnFolder.addEventListener('click', () => els.folderInput.click());
    els.folderInput.addEventListener('change', async () => {
        const files = Array.from(els.folderInput.files).filter(f => {
            const ext = f.name.split('.').pop().toLowerCase();
            return ['jpg', 'jpeg', 'png', 'bmp'].includes(ext);
        });
        if (files.length === 0) {
            showToast('info', '提示', '所选文件夹中没有图片文件');
            return;
        }
        await detectBatch(files);
        els.folderInput.value = '';
    });

    // ════════════════════════════════════════
    // 智能检测（路径直读）：粘贴文件夹路径 → 扫描结构 → 分组打勾 → 后台检测
    // 支持结构：文件夹直接是图 / 含"正面"子文件夹 / 多组（A\1-10\正面…）
    // 图片直接从硬盘读取零上传；后台任务不阻塞页面；中断可续检
    // ════════════════════════════════════════
    let scanData = null;        // 上次扫描结果
    let detectTaskId = null;    // 进行中的后台任务
    let detectCursor = 0;       // 已消费的问题图条数

    function openPathModal() {
        els.pathModal.style.display = 'flex';
        els.pathInput.value = localStorage.getItem('ct_last_path') || '';
        els.groupPanel.style.display = 'none';
        els.btnStartDetect.style.display = 'none';
        scanData = null;
    }
    function closePathModal() {
        els.pathModal.style.display = 'none';
    }
    els.btnSmart.addEventListener('click', () => {
        if (isProcessing || detectTaskId) return;
        openPathModal();
    });
    els.pathModalClose.addEventListener('click', closePathModal);
    els.pathModal.addEventListener('click', (e) => {
        if (e.target === els.pathModal) closePathModal();
    });
    els.pathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') els.btnScanPath.click();
    });

    els.btnBrowsePath.addEventListener('click', async () => {
        if (!btnBusy(els.btnBrowsePath, '选择中…')) return;
        try {
            const res = await fetch('/api/pick_folder', { method: 'POST' });
            const data = await res.json();
            if (data.success && data.path) {
                els.pathInput.value = data.path;
                els.btnScanPath.click();   // 选完自动扫描
            }
        } catch (e) {
            showToast('error', '网络错误', '打开文件夹窗口失败');
        } finally {
            btnIdle(els.btnBrowsePath);
        }
    });

    els.btnScanPath.addEventListener('click', async () => {
        const path = els.pathInput.value.trim();
        if (!path) { showToast('info', '提示', '请先粘贴文件夹路径'); return; }
        if (!btnBusy(els.btnScanPath, '扫描中…')) return;
        try {
            const res = await fetch('/api/scan_path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
            const data = await res.json();
            if (!data.success) {
                showToast('error', '扫描失败', data.message || '路径无效');
                els.groupPanel.style.display = 'none';
                els.btnStartDetect.style.display = 'none';
                return;
            }
            scanData = data;
            localStorage.setItem('ct_last_path', path);
            renderGroupList(data);
            els.btnStartDetect.style.display = 'flex';
        } catch (e) {
            showToast('error', '网络错误', '扫描请求失败');
        } finally {
            btnIdle(els.btnScanPath);
        }
    });

    function renderGroupList(data) {
        els.groupPanel.style.display = 'block';
        els.groupList.innerHTML = '';
        if (data.mode === 'groups') {
            els.groupSummary.textContent = `识别到 ${data.groups.length} 个分组，共 ${data.total} 张（打勾选择要检测的组）`;
            els.groupOps.style.display = 'inline';
            data.groups.forEach(g => {
                const row = document.createElement('label');
                row.className = 'group-item';
                row.innerHTML = `<input type="checkbox" class="group-check" value="${escapeHtml(g.name)}" checked>
                                 <span class="g-name">${escapeHtml(g.name)}</span>
                                 <span class="g-count">${g.count} 张</span>`;
                els.groupList.appendChild(row);
            });
        } else {
            const modeText = data.mode === 'flat' ? '文件夹内直接是图片' : '检测到"正面"子文件夹';
            els.groupSummary.textContent = `${modeText}，共 ${data.total} 张，直接开始即可`;
            els.groupOps.style.display = 'none';
        }
    }
    els.groupSelectAll.addEventListener('click', (e) => {
        e.preventDefault();
        els.groupList.querySelectorAll('.group-check').forEach(c => c.checked = true);
    });
    els.groupSelectNone.addEventListener('click', (e) => {
        e.preventDefault();
        els.groupList.querySelectorAll('.group-check').forEach(c => c.checked = false);
    });

    els.btnStartDetect.addEventListener('click', async () => {
        if (!scanData) return;
        let groups = null;
        if (scanData.mode === 'groups') {
            groups = [];
            els.groupList.querySelectorAll('.group-check:checked').forEach(c => groups.push(c.value));
            if (groups.length === 0) { showToast('info', '提示', '请至少勾选一个分组'); return; }
        }
        const conf = (els.confSlider.value / 100).toFixed(2);
        if (!btnBusy(els.btnStartDetect, '启动中…')) return;
        try {
            const res = await fetch('/api/detect/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: scanData.root, groups, conf })
            });
            const data = await res.json();
            if (!data.success) {
                showToast('error', '启动失败', data.message || '任务启动失败');
                return;
            }
            closePathModal();
            showToast('success', '检测已开始', `共 ${data.total} 张，后台运行中，页面可正常操作`);
            attachDetectTask(data.task_id, 0, data.batch_id);
        } catch (e) {
            showToast('error', '网络错误', '启动请求失败');
        } finally {
            btnIdle(els.btnStartDetect);
        }
    });

    // 后台任务期间：只锁检测入口与模型切换，浏览/筛选/导出/下载照常可用
    function setDetectButtons(v) {
        [els.btnPic, els.btnFolder, els.btnSmart, els.btnClear].forEach(b => {
            if (!b) return;
            b.disabled = v;
            b.style.opacity = v ? '0.5' : '';
        });
        els.modelSelect.disabled = v;
    }

    function fmtEta(s) {
        if (s < 0) return '估算中…';
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
        if (h > 0) return `${h}时${m}分`;
        if (m > 0) return `${m}分${ss}秒`;
        return `${ss}秒`;
    }

    function attachDetectTask(taskId, startCursor, batchId) {
        detectTaskId = taskId;
        detectCursor = startCursor || 0;
        // 正在看历史批次时启动/接上任务 → 主表切到实时视图
        if (batchId) switchToLive(batchId);
        els.detectFloat.style.display = 'block';
        setDetectButtons(true);
        updateBackPill();
        pollDetect();
    }

    async function pollDetect() {
        if (!detectTaskId) return;
        try {
            const st = await (await fetch(`/api/detect_status/${detectTaskId}?cursor=${detectCursor}`)).json();
            if (st.success) {
                detectCursor = st.cursor;
                const pct = st.total ? Math.round(st.done / st.total * 100) : 0;
                els.detectFloatFill.style.width = pct + '%';
                els.detectFloatStats.textContent =
                    `${st.done} / ${st.total} 张 · ${st.speed} 张/秒 · 预计剩余 ${fmtEta(st.eta_s)} · 缺陷图 ${st.defect_imgs} 张`;
                if (st.current) els.detectFloatFile.textContent = st.current.split('/').pop();

                // 新检出的问题图实时流入表格（仅当前正看着实时批次时；看历史批次不混入）
                if (st.items && st.items.length && (!viewBatchId || viewBatchId === liveBatchId)) {
                    let imageIndex = getCurrentImageCount() + 1;
                    st.items.forEach(it => { allResults.push(it); addTableRows(it, imageIndex++); });
                    animateNewRows();
                    updateRecordCount();
                    updateFilterOptions();
                    applyFilter();
                }

                if (st.status === 'done' || st.status === 'stopped' || st.status === 'error') {
                    els.detectFloat.style.display = 'none';
                    setDetectButtons(false);
                    detectTaskId = null;
                    updateBackPill();
                    if (st.status === 'done') {
                        flashScanComplete();
                        showToast('success', '检测完成', `共 ${st.total} 张，缺陷图 ${st.defect_imgs} 张 / 缺陷 ${st.defects} 个`);
                    } else if (st.status === 'stopped') {
                        showToast('info', '已停止', `完成 ${st.done}/${st.total}，重开页面可续检`);
                    } else {
                        showToast('error', '检测出错', st.message || '未知错误');
                    }
                    return;
                }
            }
        } catch (e) { /* 网络抖动，继续轮询 */ }
        setTimeout(pollDetect, 700);
    }

    els.detectStopBtn.addEventListener('click', async () => {
        if (!detectTaskId) return;
        try {
            await fetch('/api/detect/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: detectTaskId })
            });
        } catch (e) {}
    });

    async function detectBatch(files) {
        if (isProcessing) return;
        setProcessing(true);
        abortBatch = false;
        showScanLine();

        const conf = (els.confSlider.value / 100).toFixed(2);
        const batchId = 'B' + Date.now().toString(36);   // 一次检测=一个批次（所有chunk共用）
        switchToLive(batchId);   // 看历史批次时开始新检测 → 切回实时视图

        // 分批发送：每批 16 张（与后端 BATCH_SIZE 对齐），进度条按 16 张更新
        const CHUNK_SIZE = 16;
        let totalProcessed = 0;
        let totalDefectImgs = 0;
        let totalCleanImgs = 0;
        let totalFailed = 0;
        let imageIndex = getCurrentImageCount() + 1;
        let firstShown = false;

        showProgress(files.length);

        for (let chunkStart = 0; chunkStart < files.length; chunkStart += CHUNK_SIZE) {
            if (abortBatch) {
                showToast('info', '已停止', '批量检测已中断');
                break;
            }

            const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, files.length);
            const chunk = files.slice(chunkStart, chunkEnd);
            const form = new FormData();
            chunk.forEach(f => form.append('files', f));
            form.append('conf', conf);
            form.append('batch_id', batchId);
            form.append('file_paths', JSON.stringify(chunk.map(f => f.webkitRelativePath || f.name)));

            // 上传前更新进度条文字（current/total 传入右下角计数器）
            const pctBefore = Math.round(totalProcessed / files.length * 100);
            updateProgress(
                pctBefore,
                `${pctBefore}%`,
                `${pctBefore}%`,
                totalProcessed,
                files.length
            );

            try {
                const res = await fetch('/api/detect/batch', { method: 'POST', body: form });
                const data = await res.json();

                if (!data.success) {
                    showToast('error', '检测失败', data.message || '批量检测失败');
                    break;
                }

                totalProcessed += data.total || chunk.length;
                totalDefectImgs += data.ok_defect || 0;
                totalCleanImgs += data.ok_clean || 0;
                totalFailed += data.failed || 0;

                // 真实进度更新：每完成一个 chunk 就更新
                const pct = Math.round(totalProcessed / files.length * 100);
                updateProgress(
                    pct,
                    `${pct}%`,
                    `${pct}%`,
                    totalProcessed,
                    files.length
                );

                // 渲染结果（干净图后端不再画图，用本地 blob 直接显示原图）
                for (let k = 0; k < data.results.length; k++) {
                    const item = data.results[k];
                    // 用后端回传的 index 取本地 blob：绝不能用循环下标 k，
                    // 否则有文件失败时缩略图会串到别的图上（#3 的错位问题）
                    const localIdx = (typeof item.index === 'number') ? item.index : (chunkStart + k);
                    const localFile = chunk[localIdx - chunkStart];
                    if (!item.image_url && localFile) {
                        item.image_url = URL.createObjectURL(localFile);
                    }
                    const detection = {
                        filename: item.filename,
                        status: item.status || '',
                        locations: item.locations || [],
                        classes: item.classes || [],
                        confs: item.confs || [],
                        image_url: item.image_url,
                        orig_url: item.orig_url,
                        error: item.error || '',
                        time: item.time || '--'
                    };
                    if (detection.status === 'failed') {
                        // 失败项不进 allResults（它没有框/没有图），但必须出现在表格里
                        addTableRows(detection, imageIndex);
                        imageIndex += 1;
                        continue;
                    }
                    allResults.push(detection);
                    addTableRows(detection, imageIndex);
                    imageIndex += 1;

                    if (!firstShown && item.image_url) {
                        showResultImage(item.image_url, item.orig_url);
                        updateResultLabels(item.time, item.target_nums, item.confs[0], item.locations);
                        updateTargetSelect(item.classes, item.confs, item.locations);
                        firstShown = true;
                    }
                }
                animateNewRows();
                updateRecordCount();
                updateFilterOptions();
                applyFilter();

            } catch (err) {
                console.error('Batch chunk failed:', err);
                showToast('error', '网络错误', '批量请求失败');
                break;
            }
        }

        hideProgress();
        hideScanLine();
        flashScanComplete();

        if (!abortBatch) {
            const summary = `共检测 ${totalProcessed} 张：有缺陷 ${totalDefectImgs}、无缺陷 ${totalCleanImgs}、失败 ${totalFailed}`;
            if (totalFailed > 0) {
                showToast('error', '批量检测完成（存在失败）',
                          summary + '。失败项已在表格中标红，请重测后再判良品', 6000);
            } else {
                showToast('success', '批量检测完成', summary);
            }
        }
        if (!firstShown) {
            clearDisplay();
        }
        setProcessing(false);
    }

    // ── 下载所有已标注图片（打包 zip） ──
    async function downloadBlob(url, filename) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error('Download failed');
            const blob = await res.blob();
            const blobUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = blobUrl;
            a.download = filename || 'download';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(blobUrl);
        } catch (e) {
            console.error(e);
            showToast('error', '下载失败', e.message || '文件下载失败');
        }
    }

    // （旧的内联下载处理器已移除，统一走 runPackTask）

    // ── 通用：按钮 loading/防抖 ──
    const btnState = new WeakMap();
    function btnBusy(btn, text) {
        if (btnState.get(btn)) return false;   // 已在执行，忽略重复点击
        const label = btn.querySelector('span:last-child');
        btnState.set(btn, { label, orig: label ? label.textContent : '' });
        btn.disabled = true;
        btn.style.opacity = '0.6';
        btn.style.cursor = 'not-allowed';
        if (label && text) label.textContent = text;
        return true;
    }
    function btnProgress(btn, text) {
        const st = btnState.get(btn);
        if (st && st.label) st.label.textContent = text;
    }
    function btnIdle(btn) {
        const st = btnState.get(btn);
        if (!st) return;
        btn.disabled = false;
        btn.style.opacity = '';
        btn.style.cursor = '';
        if (st.label) st.label.textContent = st.orig;
        btnState.delete(btn);
    }

    // ── 通用：打包任务（导出报表/下载结果共用）：启动 → 轮询进度 → 下载 ──
    // 正在查看历史批次时，只打包当前批次；实时视图则打包全部当前结果
    async function runPackTask(btn, apiUrl, busyLabel, zipName) {
        if (!btnBusy(btn, busyLabel)) return;
        try {
            const res = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(viewBatchId ? { batch_id: viewBatchId } : {})
            });
            const data = await res.json();
            if (!data.success) {
                showToast('error', '操作失败', data.message || '任务启动失败');
                return;
            }
            const taskId = data.task_id;
            let netFails = 0;
            const deadline = Date.now() + 10 * 60 * 1000;
            await new Promise((resolve, reject) => {
                const timer = setInterval(async () => {
                    if (Date.now() > deadline) {
                        clearInterval(timer);
                        reject(new Error('打包超时，请重试'));
                        return;
                    }
                    try {
                        const st = await (await fetch(`/api/export_status/${taskId}`)).json();
                        if (!st.success) {
                            if (++netFails > 10) { clearInterval(timer); reject(new Error(st.message || '查询进度失败')); }
                            return;
                        }
                        netFails = 0;
                        if (st.total > 0) btnProgress(btn, `打包中 ${st.done}/${st.total}`);
                        else btnProgress(btn, st.message || busyLabel);
                        if (st.status === 'done') { clearInterval(timer); resolve(st); }
                        else if (st.status === 'error') { clearInterval(timer); reject(new Error(st.message)); }
                    } catch (e) {
                        if (++netFails > 10) { clearInterval(timer); reject(e); }
                    }
                }, 400);
            }).then((st) => {
                const fname = decodeURIComponent(st.url.split('/').pop());
                downloadBlob(st.url, fname);
                showToast('success', '完成', (st.message || '打包下载完成') + (viewBatchId ? '（仅当前批次）' : ''));
            });
        } catch (err) {
            console.error(err);
            showToast('error', '操作失败', err.message || '请求失败');
        } finally {
            btnIdle(btn);
        }
    }

    // ── 下载结果（问题图 zip）──
    els.btnDownload.addEventListener('click', () => {
        runPackTask(els.btnDownload, '/api/download-all', '打包中…', '问题图.zip');
    });

    // ── 导出报表（仅 Excel，同步生成直接下载；查看历史批次时只导该批）──
    els.btnExport.addEventListener('click', async () => {
        if (!btnBusy(els.btnExport, '生成报表…')) return;
        try {
            const res = await fetch('/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(viewBatchId ? { batch_id: viewBatchId } : {})
            });
            const data = await res.json();
            if (!data.success) {
                showToast('error', '导出失败', data.message || '报表导出失败');
                return;
            }
            const fname = decodeURIComponent(data.download_url.split('/').pop());
            await downloadBlob(data.download_url, fname);
            showToast('success', '导出成功', fname + (viewBatchId ? '（仅当前批次）' : ''));
        } catch (err) {
            console.error(err);
            showToast('error', '导出失败', err.message || '请求失败');
        } finally {
            btnIdle(els.btnExport);
        }
    });

    // ── 清空所有结果（弹窗确认，防误点）──
    els.btnClear.addEventListener('click', async () => {
        if (isProcessing || detectTaskId) return;
        let n = 0, m = 0;
        try {
            const st = await (await fetch('/api/state')).json();
            n = (st.detection_results || []).length;
            m = (st.problem_images || []).length;
        } catch (e) {}
        els.clearModalMsg.innerHTML = n > 0
            ? `将删除全部 <b>${n}</b> 条检测记录与 <b>${m}</b> 张问题图备份，<b>不可恢复</b>。<br>如需要请先「下载结果」或「导出报表」留存。`
            : '当前没有检测结果。';
        els.clearModal.style.display = 'flex';
    });
    function closeClearModal() { els.clearModal.style.display = 'none'; }
    els.clearCancel.addEventListener('click', closeClearModal);
    els.clearModal.addEventListener('click', (e) => { if (e.target === els.clearModal) closeClearModal(); });
    els.clearConfirm.addEventListener('click', async () => {
        closeClearModal();
        if (!btnBusy(els.btnClear, '清空中…')) return;
        try {
            await fetch('/api/clear', { method: 'POST' });
        } catch (e) {} finally {
            btnIdle(els.btnClear);
        }
        allResults = [];
        viewBatchId = null;
        liveBatchId = null;
        updateBackPill();
        els.resultTableBody.innerHTML = `
            <tr class="empty-row">
                <td colspan="6">
                    <div class="empty-state">
                        <span class="empty-icon">□</span>
                        <span class="empty-text">暂无检测数据，请先开始检测；历史批次请点「历史结果」查看</span>
                    </div>
                </td>
            </tr>`;
        clearDisplay();
        updateResultLabels(null, null, null, []);
        updateTargetSelect([], [], []);
        els.targetSelect.innerHTML = '<option value="-1">全部类别</option>';
        els.filterClass.innerHTML = '<option value="all">全部</option>';
        els.filterClass.value = 'all';
        hideProgress();
        updateRecordCount();
        showToast('info', '已清空', '所有检测结果已清除');
    });

    // ════════════════════════════════════════
    // 历史结果：按检测批次组织（一次检测=一批），可逐条/整批删除
    // ════════════════════════════════════════
    function restoreEmptyRow() {
        if (!els.resultTableBody.children.length) {
            els.resultTableBody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="6">
                        <div class="empty-state">
                            <span class="empty-icon">□</span>
                            <span class="empty-text">暂无检测数据，请先开始检测；历史批次请点「历史结果」查看</span>
                        </div>
                    </td>
                </tr>`;
        }
    }

    // 逐条删除（主表格）：该图所有检测行 + 备份文件
    async function deleteImageRecord(bid, fn) {
        if (!confirm(`删除这条记录？\n${fn}\n该图的备份文件将一并删除，不可恢复。`)) return;
        try {
            const res = await fetch('/api/state/delete_image', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ batch_id: bid, filename: fn })
            });
            const d = await res.json();
            if (d.success) {
                els.resultTableBody.querySelectorAll('tr').forEach(r => {
                    if (r.dataset.filename === fn && (r.dataset.batchId || '') === (bid || '')) r.remove();
                });
                allResults = allResults.filter(x => !(x.filename === fn && (x.batch_id || '') === (bid || '')));
                restoreEmptyRow();
                updateRecordCount();
                updateFilterOptions();
                applyFilter();
                showToast('success', '已删除', fn.split('/').pop());
            } else {
                showToast('error', '删除失败', d.message || '');
            }
        } catch (e) { showToast('error', '网络错误', '删除请求失败'); }
    }

    // 检测行+问题图 → 表格条目（一图一条，多缺陷合并）
    function buildDetections(rows, pim) {
        const pmap = {};
        (pim || []).forEach(p => {
            const rel = (p['相对目录'] ? p['相对目录'] + '/' : '') + p['文件名'];
            pmap[(p['批次ID'] || '') + '|' + rel] = {
                ann: p['标注图'] ? '/api/results/' + p['标注图'] : '',
                orig: p['原图'] ? '/api/results/' + p['原图'] : ''
            };
        });
        const groups = new Map();
        (rows || []).forEach(r => {
            const fn = r['文件路径'];
            const bid = r['批次ID'] || '';
            if (!groups.has(fn)) {
                const pm = pmap[bid + '|' + fn] || { ann: '', orig: '' };
                groups.set(fn, {
                    filename: fn, locations: [], classes: [], confs: [],
                    image_url: pm.ann, orig_url: pm.orig, batch_id: bid,
                    time: (r['检测时间'] || '--').slice(11) || '--'
                });
            }
            const d = groups.get(fn);
            d.locations.push([r['X最小值'], r['Y最小值'], r['X最大值'], r['Y最大值']]);
            d.classes.push(r['类别ID']);
            d.confs.push(r['置信度(百分比)']);
        });
        return Array.from(groups.values());
    }

    // 分批渲染条目到主表格（不卡顿）
    async function renderDetections(arr) {
        let idx = 0, imageIndex = 1;
        await new Promise(resolve => {
            function step() {
                const end = Math.min(idx + 25, arr.length);
                for (; idx < end; idx++) {
                    allResults.push(arr[idx]);
                    addTableRows(arr[idx], imageIndex++);
                }
                if (idx < arr.length) requestAnimationFrame(step);
                else resolve();
            }
            step();
        });
        updateRecordCount();
        updateFilterOptions();
        applyFilter();
    }

    // 历史批次 → 载入主界面查看
    let viewBatchId = null;   // 正在查看的批次（null=实时视图）
    let liveBatchId = null;   // 本次会话当前/最近检测的批次（"实时结果"的归属）

    // 切回实时视图：看历史批次时启动新检测 → 主表清空切到实时
    function switchToLive(bid) {
        if (bid) liveBatchId = bid;
        if (viewBatchId && viewBatchId !== liveBatchId) {
            viewBatchId = liveBatchId;
            allResults = [];
            els.resultTableBody.innerHTML = '';
            restoreEmptyRow();
            updateRecordCount();
        }
        updateBackPill();
    }

    // "返回实时结果"悬浮入口：看历史批次且与实时批次不同时出现
    function updateBackPill() {
        const show = !!(viewBatchId && liveBatchId && viewBatchId !== liveBatchId);
        els.backLivePill.style.display = show ? 'block' : 'none';
        if (show) {
            els.backLivePill.textContent = detectTaskId ? '◀ 返回正在检测的批次' : '◀ 返回实时结果';
        }
    }

    els.backLivePill.addEventListener('click', () => {
        closeHistory();
        if (liveBatchId) loadBatch(liveBatchId);
    });
    // 点浮动进度窗也可返回实时视图（停止按钮除外）
    els.detectFloat.addEventListener('click', (e) => {
        if (e.target.closest('#detectStopBtn')) return;
        closeHistory();
        if (liveBatchId && viewBatchId !== liveBatchId) loadBatch(liveBatchId);
    });

    async function loadBatch(bid) {
        showLoading('载入批次…');
        try {
            const st = await (await fetch(`/api/batch/${bid}`)).json();
            if (!st.success) { showToast('error', '读取失败', '批次数据读取失败'); return; }
            const arr = buildDetections(st.detection_results, st.problem_images);
            // #3：台账里的"检测失败"文件没有框，不在 detection_results 里，必须单独补回表格，
            // 否则历史批次回看时会以为这批"只有 N 张图"。
            const failedRows = (st.file_records || [])
                .filter(f => f['状态'] === 'failed')
                .map(f => ({
                    filename: f['文件路径'] || '', status: 'failed',
                    locations: [], classes: [], confs: [],
                    error: f['备注'] || '', batch_id: f['批次ID'] || bid, time: '--'
                }));
            viewBatchId = bid;
            allResults = [];
            els.resultTableBody.innerHTML = '';
            await renderDetections(arr.concat(failedRows));
            restoreEmptyRow();
            updateBackPill();
            const info = st.info || {};
            showToast('success', '已载入历史批次', `${info.time || ''} · 问题图 ${arr.length} 张，逐条删除直接点表格右侧 ✕`);
        } catch (e) {
            showToast('error', '网络错误', '批次读取失败');
        } finally {
            hideLoading();
        }
    }

    // 历史结果：只显示批次汇总卡片，点击载入主界面查看明细
    function renderHistory(state) {
        const rows = state.detection_results || [];
        const binfo = state.batch_info || {};
        if (!rows.length) {
            els.historyList.innerHTML = '<div class="history-empty">暂无历史结果</div>';
            return;
        }
        const batches = new Map();
        rows.forEach(r => {
            const bid = r['批次ID'] || 'L01';
            if (!batches.has(bid)) batches.set(bid, []);
            batches.get(bid).push(r);
        });
        const arr = Array.from(batches.entries())
            .map(([bid, rs]) => ({
                bid, rs,
                time: rs[0]['检测时间'] || '',
                imgs: new Set(rs.map(r => r['文件路径'])).size
            }))
            .sort((a, b) => b.time.localeCompare(a.time));
        const modeText = { path: '路径检测', groups: '路径检测', flat: '路径检测', view: '路径检测', upload: '上传检测', single: '单图检测' };

        els.historyList.innerHTML = '';
        const runningBid = detectTaskId ? liveBatchId : null;   // 正在检测的批次禁止删除（防止与后台任务打架）
        arr.forEach(b => {
            const info = binfo[b.bid] || {};
            const rootText = info.root || '—（早期数据未记录）';
            const groupsText = (info.groups && info.groups.length)
                ? info.groups.join('、')
                : (info.mode === 'single' ? '—' : '全部');
            const isRunning = b.bid === runningBid;
            // #3/#6：优先用文件台账的计数，否则"全失败"和"全干净"在卡片上长得一模一样
            const bs = (state.batch_stats || {})[b.bid];
            const statText = bs
                ? `共 ${bs.total} 张 · 有缺陷 ${bs.defect} · 无缺陷 ${bs.clean} · 失败 ${bs.failed}`
                : `问题图 ${b.imgs} 张 · 缺陷 ${b.rs.length} 个`;
            const failTag = (bs && bs.failed > 0) ? '<span class="hb-failed">有失败</span>' : '';
            const card = document.createElement('div');
            card.className = 'history-batch';
            card.innerHTML = `
                <div class="history-batch-head" title="点击载入主界面查看这一批">
                    <span class="hb-arrow">▸</span>
                    <div class="hb-info">
                        <div class="hb-title">${escapeHtml(b.time)}<span class="hb-src">${modeText[info.mode] || '检测'}</span>${isRunning ? '<span class="hb-running">检测中</span>' : ''}${failTag}</div>
                        <div class="hb-sub">路径：${escapeHtml(rootText)}</div>
                        <div class="hb-sub">分组：${escapeHtml(groupsText)} · ${statText}</div>
                    </div>
                    <button class="hb-del" ${isRunning ? 'disabled style="opacity:0.4;cursor:not-allowed;"' : ''} title="${isRunning ? '该批次正在检测中，不能删除' : '删除这一整批（含备份文件）'}">删除本批</button>
                </div>`;
            card.querySelector('.history-batch-head').addEventListener('click', () => {
                closeHistory();
                loadBatch(b.bid);
            });
            card.querySelector('.hb-del').addEventListener('click', async (e) => {
                e.stopPropagation();
                if (!confirm(`删除这一整批？\n${b.time} · 问题图 ${b.imgs} 张 · 缺陷 ${b.rs.length} 个\n备份文件一并删除，不可恢复。`)) return;
                try {
                    const res = await fetch('/api/state/delete_batch', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ batch_id: b.bid })
                    });
                    const d = await res.json();
                    if (d.success) {
                        card.remove();
                        if (b.bid === liveBatchId) liveBatchId = null;
                        if (viewBatchId === b.bid) {   // 正在看的就是这批 → 主表清空
                            viewBatchId = null;
                            allResults = [];
                            els.resultTableBody.innerHTML = '';
                            restoreEmptyRow();
                            updateRecordCount();
                            updateFilterOptions();
                            applyFilter();
                        }
                        updateBackPill();
                        showToast('success', '已删除该批', `${d.removed_rows} 条记录 / ${d.removed_imgs} 张问题图`);
                    } else {
                        showToast('error', '删除失败', d.message || '');
                    }
                } catch (err) { showToast('error', '网络错误', '删除请求失败'); }
            });
            els.historyList.appendChild(card);
        });
    }

    function closeHistory() { els.historyModal.style.display = 'none'; }
    els.btnHistory.addEventListener('click', async () => {
        els.historyModal.style.display = 'flex';
        els.historyList.innerHTML = '<div class="history-empty">读取中…</div>';
        try {
            const st = await (await fetch('/api/state')).json();
            if (st.success) renderHistory(st);
            else els.historyList.innerHTML = '<div class="history-empty">读取失败</div>';
        } catch (e) {
            els.historyList.innerHTML = '<div class="history-empty">网络错误</div>';
        }
    });
    els.historyClose.addEventListener('click', closeHistory);
    els.historyModal.addEventListener('click', (e) => { if (e.target === els.historyModal) closeHistory(); });

    // ── 全屏表格查看 ──
    function openFullscreenTable() {
        const rows = els.resultTableBody.querySelectorAll('tr:not(.empty-row)');
        if (rows.length === 0) {
            showToast('info', '提示', '暂无检测数据，无法全屏查看');
            return;
        }
        els.fullscreenTableBodyInner.innerHTML = '';
        rows.forEach(row => {
            if (row.style.display === 'none') return;
            const clone = row.cloneNode(true);
            const delCell = clone.querySelector('.cell-del');
            if (delCell) delCell.remove();   // 全屏表只读，不带删除列
            clone.addEventListener('click', () => {
                const imgUrl = clone.dataset.imageUrl || row.dataset.imageUrl;
                if (imgUrl) showResultImage(imgUrl, clone.dataset.origUrl || '');
                els.timeValue.textContent = clone.dataset.time || '--';
                els.targetCount.textContent = clone.dataset.targetCount || '--';
                els.confValue.textContent = clone.dataset.conf || '--';
                els.xmin.value = clone.dataset.xmin || '--';
                els.ymin.value = clone.dataset.ymin || '--';
                els.xmax.value = clone.dataset.xmax || '--';
                els.ymax.value = clone.dataset.ymax || '--';
            });
            els.fullscreenTableBodyInner.appendChild(clone);
        });
        els.fullscreenTableOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    function closeFullscreenTable() {
        els.fullscreenTableOverlay.style.display = 'none';
        els.fullscreenTableBodyInner.innerHTML = '';
        document.body.style.overflow = '';
    }

    els.btnExpand.addEventListener('click', (e) => {
        e.stopPropagation();
        openFullscreenTable();
    });
    els.fullscreenTableClose.addEventListener('click', closeFullscreenTable);
    els.fullscreenTableOverlay.addEventListener('click', (e) => {
        if (e.target === els.fullscreenTableOverlay) closeFullscreenTable();
    });

    // ── 全屏查看 ──
    function openFullscreen() {
        const src = els.resultImage.src;
        if (!src) return;
        els.fullscreenImage.src = src;
        els.fullscreenOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }
    function closeFullscreen() {
        els.fullscreenOverlay.style.display = 'none';
        els.fullscreenImage.src = '';
        document.body.style.overflow = '';
    }

    els.fullscreenBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        openFullscreen();
    });
    els.fullscreenClose.addEventListener('click', closeFullscreen);
    els.fullscreenOverlay.addEventListener('click', (e) => {
        if (e.target === els.fullscreenOverlay) closeFullscreen();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (els.fullscreenTableOverlay.style.display !== 'none') {
                closeFullscreenTable();
            } else if (els.fullscreenOverlay.style.display !== 'none') {
                closeFullscreen();
            }
        }
    });

    // ════════════════════════════════════════
    // 模型选择器
    // ════════════════════════════════════════
    async function loadModelList() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            if (!data.success) return;
            const select = els.modelSelect;
            select.innerHTML = '';
            data.models.forEach(m => {
                const option = document.createElement('option');
                option.value = m.id;
                option.textContent = m.label || m.name;
                if (m.active) option.selected = true;
                select.appendChild(option);
            });
            els.modelStatus.textContent = data.models.find(m => m.active)?.label || '模型就绪';
        } catch (e) {
            console.error('loadModelList failed:', e);
        }
    }

    els.modelSelect.addEventListener('change', async () => {
        const modelId = els.modelSelect.value;
        if (!modelId) return;
        const selectedName = els.modelSelect.options[els.modelSelect.selectedIndex].textContent;
        els.modelSelect.disabled = true;
        els.modelStatus.textContent = '切换中...';
        showToast('info', '切换模型', `正在加载 ${selectedName}...`);
        try {
            const res = await fetch('/api/switch_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            });
            const data = await res.json();
            if (data.success) {
                els.modelStatus.textContent = data.model_name;
                showToast('success', '切换成功', `已加载 ${data.model_name}`);
            } else {
                showToast('error', '切换失败', data.message || '模型加载失败');
                await loadModelList(); // 恢复下拉框状态
            }
        } catch (e) {
            showToast('error', '网络错误', '模型切换请求失败');
            console.error(e);
            await loadModelList();
        } finally {
            els.modelSelect.disabled = false;
        }
    });

    // ════════════════════════════════════════
    // 启动加载：分阶段进度条 → 连接服务器 → 模型列表 → 恢复历史结果（分批渲染）
    // ════════════════════════════════════════
    function bootSet(pct, text) {
        if (!els.bootOverlay || els.bootOverlay.style.display === 'none') return;
        els.bootFill.style.width = pct + '%';
        els.bootPct.textContent = pct + '%';
        if (text) els.bootStage.textContent = text;
    }
    function bootDone() {
        bootSet(100, '就绪');
        setTimeout(() => {
            els.bootOverlay.style.opacity = '0';
            setTimeout(() => { els.bootOverlay.style.display = 'none'; }, 400);
        }, 250);
    }

    async function bootInit() {
        // 1. 连接服务器（模型预热期间服务还没起来，循环重试最多90秒）
        let connected = false;
        for (let i = 0; i < 90; i++) {
            try {
                const r = await fetch('/api/models');
                if (r.ok) { connected = true; break; }
            } catch (e) {}
            bootSet(5 + Math.min(i, 55), i >= 3 ? '模型预热中，请稍候…' : '正在连接服务器…');
            await new Promise(r => setTimeout(r, 1000));
        }
        if (!connected) {
            bootSet(100, '服务器无响应，请检查启动窗口');
            return;
        }
        // 2. 模型列表
        bootSet(75, '加载模型列表…');
        await loadModelList();
        // 3. 初始界面保持空白（历史批次到「历史结果」里翻）；只处理任务恢复
        bootSet(90, '检查未完成任务…');
        let state = null;
        try { state = await (await fetch('/api/state')).json(); } catch (e) {}
        bootDone();
        if (state && state.success && state.active_task) {
            // 正在跑的任务：恢复进度条 + 把已检出的部分载入主表
            showToast('info', '检测进行中', `已接上后台任务（${state.active_task.done}/${state.active_task.total}）`);
            attachDetectTask(state.active_task.task_id, state.active_task.items_count || 0, state.active_task.batch_id);
            if (state.active_task.batch_id) loadBatch(state.active_task.batch_id);
        } else if (state && state.success && state.interrupted_task) {
            const t = state.interrupted_task;
            if (confirm(`检测到上次任务未跑完（已完成 ${t.done}/${t.total}）：\n${t.root}\n\n是否继续检测剩余部分？`)) {
                try {
                    const res = await fetch('/api/detect/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ path: t.root, conf: t.conf, resume: true })
                    });
                    const data = await res.json();
                    if (data.success) {
                        showToast('success', '续检已开始', `跳过已完成 ${data.skipped} 张，剩余 ${data.total - data.skipped} 张`);
                        attachDetectTask(data.task_id, 0, data.batch_id || t.batch_id);
                        if (t.batch_id) loadBatch(t.batch_id);
                    } else {
                        showToast('error', '续检失败', data.message || '启动失败');
                    }
                } catch (e) {
                    showToast('error', '网络错误', '续检请求失败');
                }
            }
        }
    }

    bootInit();

})();
