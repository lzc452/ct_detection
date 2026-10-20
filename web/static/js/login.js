// ── 3D 穿越星海（炫酷科技版 - 金色星星 + 深黑太空边缘）──
(function initWarpStarField() {
    const canvas = document.getElementById('starBg');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let w, h, cx, cy;
    const FOV = 250;
    const STAR_COUNT = 1200;
    const SPEED_BASE = 2.5;
    const MAX_DEPTH = 2000;
    const stars = [];

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
        cx = w / 2;
        cy = h / 2;
    }

    resize();
    window.addEventListener('resize', resize);

    class Star {
        constructor() {
            this.reset(true);
        }

        reset(randomDepth = false) {
            this.x = (Math.random() - 0.5) * w * 4;
            this.y = (Math.random() - 0.5) * h * 4;
            this.z = randomDepth ? Math.random() * MAX_DEPTH : MAX_DEPTH;
            this.size = Math.random() * 3 + 1;
            this.hue = Math.random() * 25 + 35; // 35-60 金色暖色调
            this.speed = Math.random() * 3 + SPEED_BASE;
            this.prevX = null;
            this.prevY = null;
        }

        update() {
            const scale = FOV / Math.max(this.z, 0.01);
            this.prevX = this.x * scale + cx;
            this.prevY = this.y * scale + cy;
            this.z -= this.speed;
            if (this.z <= 0.1) {
                this.reset();
            }
        }

        draw() {
            const scale = FOV / this.z;
            const x = this.x * scale + cx;
            const y = this.y * scale + cy;
            const proximity = 1 - this.z / MAX_DEPTH;
            const brightness = Math.min(1, proximity * 2.5);
            const size = this.size * scale * 0.4;

            // 尾迹 - 金色渐变
            if (this.prevX !== null && this.z < MAX_DEPTH * 0.9) {
                const dx = x - this.prevX;
                const dy = y - this.prevY;
                const velocity = Math.hypot(dx, dy);
                const tailLen = Math.min(200, velocity * 5);
                const angle = Math.atan2(dy, dx);
                const tailX = x - Math.cos(angle) * tailLen;
                const tailY = y - Math.sin(angle) * tailLen;

                const grad = ctx.createLinearGradient(tailX, tailY, x, y);
                grad.addColorStop(0, `hsla(${this.hue}, 80%, 50%, 0)`);
                grad.addColorStop(0.4, `hsla(${this.hue}, 90%, 60%, ${brightness * 0.5})`);
                grad.addColorStop(1, `hsla(45, 100%, 80%, ${brightness * 0.9})`);

                ctx.beginPath();
                ctx.moveTo(tailX, tailY);
                ctx.lineTo(x, y);
                ctx.strokeStyle = grad;
                ctx.lineWidth = Math.max(1, size * 1.2);
                ctx.lineCap = 'round';
                ctx.stroke();
            }

            // 星星本体 - 亮白金色
            ctx.beginPath();
            ctx.arc(x, y, Math.max(1, size), 0, Math.PI * 2);
            ctx.fillStyle = `hsla(50, 100%, 90%, ${brightness})`;
            ctx.fill();

            // 双层金色光晕
            if (brightness > 0.4) {
                ctx.beginPath();
                ctx.arc(x, y, size * 4, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${this.hue}, 80%, 60%, ${brightness * 0.35})`;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(x, y, size * 8, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${this.hue}, 60%, 40%, ${brightness * 0.12})`;
                ctx.fill();
            }
        }
    }

    for (let i = 0; i < STAR_COUNT; i++) {
        stars.push(new Star());
    }

    function drawBg() {
        // 太空背景：中心深蓝，边缘纯黑
        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.9);
        gradient.addColorStop(0, '#0a1e4a');
        gradient.addColorStop(0.2, '#061430');
        gradient.addColorStop(0.45, '#030a18');
        gradient.addColorStop(0.7, '#01040a');
        gradient.addColorStop(1, '#000000');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, w, h);

        // 中心微弱的金色引擎辉光
        const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 350);
        glow.addColorStop(0, 'rgba(40, 80, 160, 0.08)');
        glow.addColorStop(0.5, 'rgba(20, 40, 90, 0.03)');
        glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = glow;
        ctx.fillRect(0, 0, w, h);
    }

    let animId;
    function loop() {
        drawBg();
        for (let star of stars) {
            star.update();
            star.draw();
        }
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

// ── 登录验证 ──
(function() {
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const errorMsg = document.getElementById('errorMsg');
    const loginBtn = document.getElementById('loginBtn');

    const VALID_USER = 'HPT';
    const VALID_PASS = '123456';

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.add('show');
        errorMsg.classList.remove('shake');
        void errorMsg.offsetWidth; // 强制重绘
        errorMsg.classList.add('shake');
    }

    function clearError() {
        errorMsg.textContent = '';
        errorMsg.classList.remove('show');
    }

    function doLogin() {
        const user = usernameInput.value.trim();
        const pass = passwordInput.value.trim();
        clearError();

        if (!user) {
            showError('请输入账号');
            usernameInput.focus();
            return;
        }
        if (!pass) {
            showError('请输入密码');
            passwordInput.focus();
            return;
        }
        if (user !== VALID_USER || pass !== VALID_PASS) {
            showError('密码错误');
            passwordInput.value = '';
            passwordInput.focus();
            return;
        }

        // 验证成功
        loginBtn.innerHTML = '<span class="btn-icon">✓</span><span>登录成功</span>';
        loginBtn.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';

        // 模拟跳转延迟
        setTimeout(() => {
            window.location.href = '/main';
        }, 800);
    }

    loginBtn.addEventListener('click', doLogin);
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        doLogin();
    });

    // 回车登录（最可靠的方式）
    loginForm.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            doLogin();
        }
    });

    // 眼睛按钮：显示/隐藏密码
    const eyeBtn = document.getElementById('eyeBtn');
    const eyeIcon = document.getElementById('eyeIcon');
    if (eyeBtn) {
        eyeBtn.addEventListener('click', () => {
            const isPwd = passwordInput.type === 'password';
            passwordInput.type = isPwd ? 'text' : 'password';
            eyeIcon.textContent = isPwd ? '◎' : '◉';
            eyeBtn.classList.toggle('active', isPwd);
            eyeBtn.title = isPwd ? '隐藏密码' : '显示密码';
        });
    }

    // 回车切换焦点（账号框回车跳到密码框）
    usernameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            passwordInput.focus();
        }
    });
})();
