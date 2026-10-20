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
        fullscreenOverlay: document.getElementById('fullscreenOverlay'),
        fullscreenImage: document.getElementById('fullscreenImage'),
        fullscreenClose: document.getElementById('fullscreenClose'),
        filterClass: document.getElementById('filterClass'),
        filterHasConf: document.getElementById('filterHasConf'),
        btnExpand: document.getElementById('btnExpand'),
        fullscreenTableOverlay: document.getElementById('fullscreenTableOverlay'),
        fullscreenTableBodyInner: document.getElementById('fullscreenTableBodyInner'),
        fullscreenTableClose: document.getElementById('fullscreenTableClose'),
        modelSelect: document.getElementById('modelSelect'),
        modelStatus: document.getElementById('modelStatus'),
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

    // 参数同步
    els.confSlider.addEventListener('input', () => {
        const v = els.confSlider.value / 100;
        els.confDisplay.textContent = v.toFixed(2);
    });

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
        currentMode = 'none';
    }

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
        const time = detection.time || '--';
        const hasTargets = locations.length > 0;

        if (!hasTargets) {
            const row = document.createElement('tr');
            row.className = 'no-target';
            row.innerHTML = `<td colspan="5"><span style="color:#8A8580;">${imageIndex} - 未检出目标：${escapeHtml(filename)}</span></td>`;
            row.dataset.imageUrl = imageUrl;
            row.dataset.time = time;
            row.dataset.targetCount = 0;
            row.dataset.filename = filename;
            row.dataset.imageIndex = imageIndex;
            row.addEventListener('click', () => {
                highlightRow(row);
                if (row.dataset.imageUrl) showImage(row.dataset.imageUrl);
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
            row.dataset.filename = filename;
            row.dataset.imageIndex = imageIndex;
            row.dataset.defectIndex = idx;

            row.addEventListener('click', () => {
                highlightRow(row);
                const imgUrl = row.dataset.imageUrl;
                if (imgUrl) showImage(imgUrl);

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
        const hasConfOnly = els.filterHasConf.checked;
        const rows = els.resultTableBody.querySelectorAll('tr');

        let visibleCount = 0;
        rows.forEach(row => {
            if (row.classList.contains('empty-row')) return;

            const targetCount = parseInt(row.dataset.targetCount || '0');
            const clsName = row.dataset.clsName || '';

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
    els.filterHasConf.addEventListener('change', applyFilter);

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
        form.append('file_paths', JSON.stringify([file.webkitRelativePath || file.name]));

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

            showImage(data.image_url);
            updateResultLabels(data.time, data.target_nums, data.confs[0], data.locations);
            updateTargetSelect(data.classes, data.confs, data.locations);

            const imageIndex = getCurrentImageCount() + 1;
            const detection = {
                filename: data.filename,
                locations: data.locations,
                classes: data.classes,
                confs: data.confs,
                image_url: data.image_url,
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
                    row.innerHTML = `<td colspan="5" style="color:#A0834B; font-weight:500; text-align:center; padding:10px;">Video processed: ${escapeHtml(file.name)}</td>`;
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
    // 智能检测：选母文件夹（图片合集），自动只测各电芯的"正面"图
    // 目录结构：图片合集/A/正面|侧面|上面/xxx.jpg，报表按 A/B/C 分组
    // ════════════════════════════════════════
    const SMART_TARGET_VIEW = '正面';   // 只检测这个视角
    els.btnSmart.addEventListener('click', () => els.smartInput.click());
    els.smartInput.addEventListener('change', async () => {
        const all = Array.from(els.smartInput.files);
        const isImg = f => ['jpg', 'jpeg', 'png', 'bmp'].includes(f.name.split('.').pop().toLowerCase());
        const frontFiles = all.filter(f => {
            if (!isImg(f)) return false;
            const parts = (f.webkitRelativePath || '').split('/');
            return parts.includes(SMART_TARGET_VIEW);
        });
        if (frontFiles.length === 0) {
            showToast('error', '未找到正面图',
                `智能检测要求目录结构为：合集/电芯编号/${SMART_TARGET_VIEW}/图片。请确认选择的是母文件夹。`);
            els.smartInput.value = '';
            return;
        }
        const groups = new Set(frontFiles.map(f => {
            const parts = (f.webkitRelativePath || '').split('/');
            return parts[parts.indexOf(SMART_TARGET_VIEW) - 1] || '';
        }));
        showToast('info', '智能检测', `识别到 ${groups.size} 个组（${Array.from(groups).join('、')}），共 ${frontFiles.length} 张正面图`);
        await detectBatch(frontFiles);
        els.smartInput.value = '';
    });

    async function detectBatch(files) {
        if (isProcessing) return;
        setProcessing(true);
        abortBatch = false;
        showScanLine();

        const conf = (els.confSlider.value / 100).toFixed(2);

        // 分批发送：每批 16 张（与后端 BATCH_SIZE 对齐），进度条按 16 张更新
        const CHUNK_SIZE = 16;
        let totalProcessed = 0;
        let totalDefects = 0;
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
                totalDefects += data.results.filter(r => r.target_nums > 0).length;

                // 真实进度更新：每完成一个 chunk 就更新
                const pct = Math.round(totalProcessed / files.length * 100);
                updateProgress(
                    pct,
                    `${pct}%`,
                    `${pct}%`,
                    totalProcessed,
                    files.length
                );

                // 渲染结果
                for (const item of data.results) {
                    const detection = {
                        filename: item.filename,
                        locations: item.locations,
                        classes: item.classes,
                        confs: item.confs,
                        image_url: item.image_url,
                        time: item.time || '--'
                    };
                    allResults.push(detection);
                    addTableRows(detection, imageIndex);
                    imageIndex += 1;

                    if (!firstShown && item.image_url) {
                        showImage(item.image_url);
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
            showToast('success', '批量检测完成', `共检测 ${totalProcessed} 张，有缺陷 ${totalDefects} 张`);
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
    async function runPackTask(btn, apiUrl, busyLabel, zipName) {
        if (!btnBusy(btn, busyLabel)) return;
        try {
            const res = await fetch(apiUrl, { method: 'POST' });
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
                showToast('success', '完成', st.message || '打包下载完成');
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

    // ── 导出报表（仅 Excel，同步生成直接下载）──
    els.btnExport.addEventListener('click', async () => {
        if (!btnBusy(els.btnExport, '生成报表…')) return;
        try {
            const res = await fetch('/api/export', { method: 'POST' });
            const data = await res.json();
            if (!data.success) {
                showToast('error', '导出失败', data.message || '报表导出失败');
                return;
            }
            const fname = decodeURIComponent(data.download_url.split('/').pop());
            await downloadBlob(data.download_url, fname);
            showToast('success', '导出成功', fname);
        } catch (err) {
            console.error(err);
            showToast('error', '导出失败', err.message || '请求失败');
        } finally {
            btnIdle(els.btnExport);
        }
    });

    // ── 清空结果 ──
    els.btnClear.addEventListener('click', async () => {
        if (!confirm('确定要清空所有检测结果吗？')) return;
        if (!btnBusy(els.btnClear, '清空中…')) return;
        try {
            await fetch('/api/clear', { method: 'POST' });
        } catch (e) {} finally {
            btnIdle(els.btnClear);
        }
        allResults = [];
        els.resultTableBody.innerHTML = `
            <tr class="empty-row">
                <td colspan="5">
                    <div class="empty-state">
                        <span class="empty-icon">□</span>
                        <span class="empty-text">暂无检测数据，请先上传影像</span>
                    </div>
                </td>
            </tr>`;
        clearDisplay();
        updateResultLabels(null, null, null, []);
        updateTargetSelect([], [], []);
        els.targetSelect.innerHTML = '<option value="-1">全部类别</option>';
        els.filterClass.innerHTML = '<option value="all">全部</option>';
        els.filterClass.value = 'all';
        els.filterHasConf.checked = true;
        hideProgress();
        updateRecordCount();
        showToast('info', '已清空', '所有检测结果已清除');
    });

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
            clone.addEventListener('click', () => {
                const imgUrl = clone.dataset.imageUrl || row.dataset.imageUrl;
                if (imgUrl) showImage(imgUrl);
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

    // 页面加载时获取模型列表
    loadModelList();

})();
