/**
 * Foot Size Finder — Desktop GUI Script
 * Auth: Firebase | Nav: Sidebar sections | Backend: FastAPI cv_pipeline
 */
document.addEventListener('DOMContentLoaded', () => {

    // ── Pages ──
    const loginPage = document.getElementById('loginPage');
    const mainApp   = document.getElementById('mainApp');

    // ── Sections ──
    const sectionDashboard = document.getElementById('sectionDashboard');
    const sectionScanner   = document.getElementById('sectionScanner');
    const sectionHistory   = document.getElementById('sectionHistory');

    // ── Nav ──
    const navItems = document.querySelectorAll('.nav-item');

    // ── Scanner elements ──
    const videoElement    = document.getElementById('cameraFeed');
    const captureCanvas   = document.getElementById('captureCanvas');
    const cameraFallbackBg = document.getElementById('cameraFallbackBg');
    const bracketOverlay  = document.getElementById('bracketOverlay');
    const bracketLeft     = document.getElementById('bracketLeft');
    const bracketRight    = document.getElementById('bracketRight');
    const toggleLeft      = document.getElementById('toggleLeft');
    const toggleRight     = document.getElementById('toggleRight');
    const captureBtn      = document.getElementById('captureBtn');
    const autoCaptureToggle = document.getElementById('autoCaptureToggle');
    const photoUpload     = document.getElementById('photoUpload');
    const pipCanvas       = document.getElementById('pipCanvas');
    const processingCard  = document.getElementById('processingCard');
    const resultPreviewCard = document.getElementById('resultPreviewCard');
    const processingStatus  = document.getElementById('processingStatus');
    const pStepsSm        = document.querySelectorAll('.p-step-sm');

    // ── Dashboard ──
    const statTotal    = document.getElementById('statTotal');
    const statLastSize = document.getElementById('statLastSize');
    const statLastDate = document.getElementById('statLastDate');
    const recentCards  = document.getElementById('recentCards');
    const welcomeTitle = document.getElementById('welcomeTitle');
    const dashScanBtn  = document.getElementById('dashScanBtn');
    const viewAllBtn   = document.getElementById('viewAllBtn');

    // ── History ──
    const historyTableBody = document.getElementById('historyTableBody');
    const clearHistoryBtn  = document.getElementById('clearHistoryBtn');

    // ── User UI ──
    const sidebarAvatar   = document.getElementById('sidebarAvatar');
    const sidebarUserName = document.getElementById('sidebarUserName');
    const topbarAvatar    = document.getElementById('topbarAvatar');
    const topbarName      = document.getElementById('topbarName');
    const topbarTitle     = document.getElementById('topbarTitle');
    const signOutBtn      = document.getElementById('signOutBtn');

    // ── Mobile UI ──
    const mobileMenuBtn   = document.getElementById('mobileMenuBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const sidebarOverlay  = document.getElementById('sidebarOverlay');
    const sidebar         = document.getElementById('sidebar');

    // ── State ──
    const API_BASE  = window.API_BASE_URL || 'http://localhost:8000';
    let currentUser = null;
    let currentFoot = 'right';
    let currentStream = null;
    let cameraReady   = false;
    let isCapturing   = false;
    let pipAnimFrame  = null;
    let autoCaptureEnabled = false;
    let targetAccel = null;
    let stableStartTime = 0;

    // ═══════════════════════════════════════
    // PARTICLE BACKGROUND (login)
    // ═══════════════════════════════════════
    (function initParticles() {
        const canvas = document.getElementById('loginCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
        resize(); window.addEventListener('resize', resize);
        const pts = Array.from({length: 70}, () => ({
            x: Math.random() * canvas.width, y: Math.random() * canvas.height,
            r: Math.random() * 1.6 + 0.4,
            dx: (Math.random()-.5)*.35, dy: (Math.random()-.5)*.35,
            a: Math.random()*.5+.1
        }));
        function draw() {
            ctx.clearRect(0,0,canvas.width,canvas.height);
            pts.forEach(p => {
                ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
                ctx.fillStyle=`rgba(41,197,246,${p.a})`; ctx.fill();
                p.x+=p.dx; p.y+=p.dy;
                if(p.x<0)p.x=canvas.width; if(p.x>canvas.width)p.x=0;
                if(p.y<0)p.y=canvas.height; if(p.y>canvas.height)p.y=0;
            });
            for(let i=0;i<pts.length;i++) for(let j=i+1;j<pts.length;j++){
                const d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);
                if(d<90){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);
                ctx.strokeStyle=`rgba(41,197,246,${.06*(1-d/90)})`;ctx.lineWidth=.5;ctx.stroke();}
            }
            requestAnimationFrame(draw);
        }
        draw();
    })();

    // ═══════════════════════════════════════
    // AUTH HELPERS
    // ═══════════════════════════════════════
    function showAuthError(msg) {
        const el = document.getElementById('authError');
        if (el) el.textContent = msg;
    }
    function clearAuthError() {
        const el = document.getElementById('authError');
        if (el) el.textContent = '';
    }

    function onAuthSuccess(firebaseUser) {
        currentUser = {
            name:  firebaseUser.displayName || firebaseUser.email.split('@')[0],
            email: firebaseUser.email,
            uid:   firebaseUser.uid
        };
        const initials = currentUser.name.split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);
        [sidebarAvatar, topbarAvatar].forEach(el => { if(el) el.textContent = initials; });
        if(sidebarUserName) sidebarUserName.textContent = currentUser.name;
        if(topbarName) topbarName.textContent = currentUser.name;
        if(welcomeTitle) welcomeTitle.textContent = `Welcome, ${currentUser.name.split(' ')[0]}!`;
        loginPage.classList.add('hidden');
        mainApp.classList.remove('hidden');
        refreshDashboard();
    }

    // Tab switcher
    const tabSignIn   = document.getElementById('tabSignIn');
    const tabSignUp   = document.getElementById('tabSignUp');
    const panelSignIn = document.getElementById('panelSignIn');
    const panelSignUp = document.getElementById('panelSignUp');

    if(tabSignIn) tabSignIn.addEventListener('click', () => {
        tabSignIn.classList.add('active'); tabSignUp.classList.remove('active');
        panelSignIn.classList.remove('hidden'); panelSignUp.classList.add('hidden');
        clearAuthError();
        const h = document.querySelector('.auth-title');
        if(h) h.textContent = 'Welcome back';
    });
    if(tabSignUp) tabSignUp.addEventListener('click', () => {
        tabSignUp.classList.add('active'); tabSignIn.classList.remove('active');
        panelSignUp.classList.remove('hidden'); panelSignIn.classList.add('hidden');
        clearAuthError();
        const h = document.querySelector('.auth-title');
        if(h) h.textContent = 'Create account';
    });

    // Google sign-in
    ['googleSignInBtn','googleSignUpBtn'].forEach(id => {
        const btn = document.getElementById(id);
        if(btn && window.fsf) btn.addEventListener('click', async () => {
            clearAuthError(); btn.disabled = true; btn.textContent = 'Signing in...';
            try { const c = await window.fsf.signInWithGoogle(); onAuthSuccess(c.user); }
            catch(e) { showAuthError(e.message||'Google sign-in failed.'); btn.disabled=false; btn.textContent='Continue with Google'; }
        });
    });

    // Email sign-in
    const signInForm = document.getElementById('signInForm');
    if(signInForm && window.fsf) signInForm.addEventListener('submit', async e => {
        e.preventDefault(); clearAuthError();
        const email = document.getElementById('signInEmail').value.trim();
        const pass  = document.getElementById('signInPassword').value;
        const btn   = document.getElementById('signInBtn');
        btn.disabled=true; btn.textContent='Signing in...';
        try { const c = await window.fsf.signInWithEmail(email,pass); onAuthSuccess(c.user); }
        catch(e) { showAuthError(
            e.code==='auth/wrong-password'?'Incorrect password.':
            e.code==='auth/user-not-found'?'No account with this email.':
            e.message||'Sign-in failed.');
        } finally { btn.disabled=false; btn.textContent='SIGN IN'; }
    });

    // Create account
    const signUpForm = document.getElementById('signUpForm');
    if(signUpForm && window.fsf) signUpForm.addEventListener('submit', async e => {
        e.preventDefault(); clearAuthError();
        const name  = document.getElementById('signUpName').value.trim();
        const email = document.getElementById('signUpEmail').value.trim();
        const pass  = document.getElementById('signUpPassword').value;
        const btn   = document.getElementById('signUpBtn');
        btn.disabled=true; btn.textContent='Creating...';
        try { const c = await window.fsf.createAccount(name,email,pass); onAuthSuccess(c.user); }
        catch(e) { showAuthError(
            e.code==='auth/email-already-in-use'?'Email already registered.':
            e.code==='auth/weak-password'?'Password must be at least 6 characters.':
            e.message||'Could not create account.');
        } finally { btn.disabled=false; btn.textContent='CREATE ACCOUNT'; }
    });

    // Restore session
    if(window.fsf) window.fsf.onAuthChanged(user => {
        if(user && loginPage && !loginPage.classList.contains('hidden')) onAuthSuccess(user);
    });

    // Sign out
    const topbarSignOutBtn = document.getElementById('topbarSignOutBtn');

    function handleSignOut() {
        if(window.fsf) window.fsf.signOut();
        stopCamera(); currentUser = null;
        mainApp.classList.add('hidden'); loginPage.classList.remove('hidden');
    }

    if(signOutBtn) signOutBtn.addEventListener('click', handleSignOut);
    if(topbarSignOutBtn) topbarSignOutBtn.addEventListener('click', handleSignOut);

    // ═══════════════════════════════════════
    // SIDEBAR NAVIGATION
    // ═══════════════════════════════════════
    const sections = { dashboard: sectionDashboard, scanner: sectionScanner, history: sectionHistory };
    const titles   = { dashboard: 'Dashboard', scanner: 'Scanner', history: 'Scan History' };

    function closeSidebar() {
        if(sidebar) sidebar.classList.remove('open');
        if(sidebarOverlay) sidebarOverlay.classList.remove('active');
    }

    if(mobileMenuBtn) mobileMenuBtn.addEventListener('click', () => {
        if(sidebar) sidebar.classList.add('open');
        if(sidebarOverlay) sidebarOverlay.classList.add('active');
    });
    if(closeSidebarBtn) closeSidebarBtn.addEventListener('click', closeSidebar);
    if(sidebarOverlay) sidebarOverlay.addEventListener('click', closeSidebar);

    function showSection(name) {
        Object.values(sections).forEach(s => s.classList.remove('active'));
        sections[name].classList.add('active');
        navItems.forEach(n => n.classList.toggle('active', n.dataset.section === name));
        if(topbarTitle) topbarTitle.textContent = titles[name];
        closeSidebar();
        if(name === 'scanner') {
            if(!cameraReady) initCamera();
        } else {
            stopCamera();
        }
        if(name === 'dashboard') refreshDashboard();
        if(name === 'history')   renderHistoryTable();
    }

    navItems.forEach(btn => btn.addEventListener('click', () => showSection(btn.dataset.section)));
    if(dashScanBtn) dashScanBtn.addEventListener('click', () => showSection('scanner'));
    if(viewAllBtn)  viewAllBtn.addEventListener('click',  () => showSection('history'));

    // ═══════════════════════════════════════
    // HISTORY STORAGE
    // ═══════════════════════════════════════
    async function getHistory() {
        try {
            return window.fsf ? await window.fsf.getHistory() :
                JSON.parse(localStorage.getItem('fsf_history_guest')||'[]');
        } catch { return []; }
    }

    async function saveResult(result) {
        const entry = {
            id: Date.now(), timestamp: new Date().toISOString(),
            foot: result.foot, length_cm: result.length_cm, width_cm: result.width_cm,
            uk: result.uk, us: result.us, eu: result.eu, ind: result.ind
        };
        if(window.fsf) await window.fsf.saveResult(entry);
        else {
            const arr = JSON.parse(localStorage.getItem('fsf_history_guest')||'[]');
            arr.unshift(entry);
            localStorage.setItem('fsf_history_guest', JSON.stringify(arr.slice(0,30)));
        }
        return entry;
    }

    // ═══════════════════════════════════════
    // DASHBOARD
    // ═══════════════════════════════════════
    async function refreshDashboard() {
        const history = await getHistory();
        if(statTotal) statTotal.textContent = history.length;
        if(history.length > 0) {
            const last = history[0];
            const d = new Date(last.timestamp);
            if(statLastSize) statLastSize.textContent = `${last.ind}`;
            if(statLastDate) statLastDate.textContent = d.toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'});
        }
        renderRecentCards(history.slice(0,3));
    }

    function renderRecentCards(items) {
        if(!recentCards) return;
        if(!items.length) {
            recentCards.innerHTML = `<div class="empty-state"><div class="empty-icon">📷</div><p>No scans yet. Start scanning to see results here.</p></div>`;
            return;
        }
        recentCards.innerHTML = items.map(e => {
            const d  = new Date(e.timestamp);
            const dt = d.toLocaleDateString('en-IN',{day:'2-digit',month:'short'}) + ' · ' +
                       d.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',hour12:true});
            return `<div class="recent-card">
                <div class="recent-card-foot">${e.foot.toUpperCase()} FOOT</div>
                <div class="recent-card-size">${e.ind}</div>
                <div class="recent-card-sub">${e.length_cm} cm × ${e.width_cm} cm &nbsp;·&nbsp; US ${e.us} / EU ${e.eu}</div>
                <div class="recent-card-time">${dt}</div>
            </div>`;
        }).join('');
    }

    // ═══════════════════════════════════════
    // HISTORY TABLE
    // ═══════════════════════════════════════
    async function renderHistoryTable() {
        const history = await getHistory();
        if(!historyTableBody) return;
        if(!history.length) {
            historyTableBody.innerHTML = `<tr class="empty-row"><td colspan="9">No scan records yet. Go to Scanner to get started.</td></tr>`;
            return;
        }
        historyTableBody.innerHTML = history.map((e,i) => {
            const d = new Date(e.timestamp);
            const dateStr = d.toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'});
            const timeStr = d.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',hour12:true});
            return `<tr>
                <td>${i+1}</td>
                <td>${dateStr}</td>
                <td>${timeStr}</td>
                <td><span class="foot-badge ${e.foot}">${e.foot}</span></td>
                <td>${e.length_cm} cm</td>
                <td>${e.width_cm} cm</td>
                <td class="size-val">${e.ind}</td>
                <td class="size-val">${e.us}</td>
                <td class="size-val">${e.eu}</td>
                <td>
                    <button class="delete-row-btn" data-id="${e.id}" title="Delete Record">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                    </button>
                </td>
            </tr>`;
        }).join('');
    }

    if(clearHistoryBtn) clearHistoryBtn.addEventListener('click', async () => {
        if(!confirm('Clear all scan history?')) return;
        if(window.fsf) {
            await window.fsf.clearHistory();
        } else {
            localStorage.removeItem('fsf_history_guest');
        }
        await renderHistoryTable();
        await refreshDashboard();
    });

    if(historyTableBody) historyTableBody.addEventListener('click', async e => {
        const btn = e.target.closest('.delete-row-btn');
        if(!btn) return;
        const id = btn.dataset.id;
        if(!confirm('Delete this scan record?')) return;
        if(window.fsf && window.fsf.deleteResult) {
            await window.fsf.deleteResult(id);
        } else {
            // Fallback for guest
            const arr = JSON.parse(localStorage.getItem('fsf_history_guest')||'[]');
            const updated = arr.filter(item => item.id.toString() !== id.toString());
            localStorage.setItem('fsf_history_guest', JSON.stringify(updated));
        }
        await renderHistoryTable();
        await refreshDashboard();
    });

    // ═══════════════════════════════════════
    // CAMERA
    // ═══════════════════════════════════════
    function startPipMirror() {
        if(!pipCanvas||!videoElement) return;
        const ctx = pipCanvas.getContext('2d');
        function frame() {
            if(videoElement && !videoElement.paused && videoElement.readyState>=2) {
                pipCanvas.width=videoElement.videoWidth||320; pipCanvas.height=videoElement.videoHeight||240;
                ctx.drawImage(videoElement,0,0,pipCanvas.width,pipCanvas.height);
            }
            pipAnimFrame = requestAnimationFrame(frame);
        }
        frame();
    }

    function stopPipMirror() {
        if(pipAnimFrame){cancelAnimationFrame(pipAnimFrame);pipAnimFrame=null;}
    }

    async function initCamera() {
        if(!videoElement) return;
        videoElement.style.opacity = '1';
        if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){showFallback();return;}
        try {
            const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1920},height:{ideal:1080}}});
            videoElement.srcObject = stream; currentStream = stream; cameraReady = true;
            if(cameraFallbackBg) cameraFallbackBg.classList.remove('active');
            videoElement.addEventListener('playing', startPipMirror, {once:true});
        } catch {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({video:true});
                videoElement.srcObject = stream; currentStream = stream; cameraReady = true;
                if(cameraFallbackBg) cameraFallbackBg.classList.remove('active');
                videoElement.addEventListener('playing', startPipMirror, {once:true});
            } catch { showFallback(); }
        }
    }

    function stopCamera() {
        stopPipMirror();
        if(currentStream){currentStream.getTracks().forEach(t=>t.stop());currentStream=null;}
        cameraReady = false;
    }

    function showFallback() {
        cameraReady = false;
        if(videoElement) videoElement.style.display = 'none';
        if(cameraFallbackBg) cameraFallbackBg.classList.add('active');
    }

    // ═══════════════════════════════════════
    // FOOT TOGGLE
    // ═══════════════════════════════════════
    function setActiveFoot(side) {
        currentFoot = side;
        [toggleLeft,toggleRight].forEach(b=>{if(b)b.classList.toggle('active',b.dataset.side===side);});
        if(bracketOverlay){
            bracketOverlay.classList.toggle('side-left', side==='left');
            bracketOverlay.classList.toggle('side-right', side==='right');
        }
    }
    if(toggleLeft)  toggleLeft.addEventListener('click',  ()=>setActiveFoot('left'));
    if(toggleRight) toggleRight.addEventListener('click', ()=>setActiveFoot('right'));

    // ═══════════════════════════════════════
    // AUTO-CAPTURE (GYROSCOPE)
    // ═══════════════════════════════════════
    function handleMotion(event) {
        if (!autoCaptureEnabled || !cameraReady || isCapturing) {
            targetAccel = null;
            if (captureBtn && captureBtn.textContent === 'HOLD STILL...') {
                captureBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg> CAPTURE';
            }
            return;
        }

        const accel = event.accelerationIncludingGravity;
        if (!accel || accel.x === null) return;

        if (!targetAccel) {
            targetAccel = { x: accel.x, y: accel.y, z: accel.z };
            stableStartTime = Date.now();
            return;
        }

        const dx = accel.x - targetAccel.x;
        const dy = accel.y - targetAccel.y;
        const dz = accel.z - targetAccel.z;
        const delta = Math.sqrt(dx*dx + dy*dy + dz*dz);

        if (delta > 0.8) { // Moved too much
            targetAccel = { x: accel.x, y: accel.y, z: accel.z };
            stableStartTime = Date.now();
            if (captureBtn && captureBtn.textContent === 'HOLD STILL...') {
                captureBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg> CAPTURE';
            }
        } else {
            const elapsed = Date.now() - stableStartTime;
            if (elapsed > 400 && captureBtn && captureBtn.textContent !== 'HOLD STILL...') {
                captureBtn.textContent = 'HOLD STILL...';
            }
            if (elapsed > 1500) {
                targetAccel = null;
                // Disable Auto-Capture immediately so it doesn't loop infinitely
                autoCaptureEnabled = false;
                if (autoCaptureToggle) {
                    autoCaptureToggle.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22A10 10 0 1 0 12 2a10 10 0 0 0 0 20z"></path><path d="M12 6v6l4 2"></path></svg> AUTO: OFF';
                    autoCaptureToggle.style.color = '';
                    autoCaptureToggle.style.borderColor = '';
                }
                window.removeEventListener('devicemotion', handleMotion);
                if (captureBtn) captureBtn.click();
            }
        }
    }

    if (autoCaptureToggle) {
        autoCaptureToggle.addEventListener('click', async () => {
            if (!autoCaptureEnabled) {
                if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
                    try {
                        const permission = await DeviceMotionEvent.requestPermission();
                        if (permission !== 'granted') {
                            alert('Motion permission is required for Auto-Capture.');
                            return;
                        }
                    } catch (e) {
                        console.error(e);
                        alert('Could not request motion permission.');
                        return;
                    }
                }
                autoCaptureEnabled = true;
                autoCaptureToggle.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22A10 10 0 1 0 12 2a10 10 0 0 0 0 20z"></path><path d="M12 6v6l4 2"></path></svg> AUTO: ON';
                autoCaptureToggle.style.color = '#34D399';
                autoCaptureToggle.style.borderColor = 'rgba(52,211,153,0.3)';
                window.addEventListener('devicemotion', handleMotion);
            } else {
                autoCaptureEnabled = false;
                targetAccel = null;
                autoCaptureToggle.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22A10 10 0 1 0 12 2a10 10 0 0 0 0 20z"></path><path d="M12 6v6l4 2"></path></svg> AUTO: OFF';
                autoCaptureToggle.style.color = '';
                autoCaptureToggle.style.borderColor = '';
                window.removeEventListener('devicemotion', handleMotion);
                if (captureBtn && captureBtn.textContent === 'HOLD STILL...') {
                    captureBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg> CAPTURE';
                }
            }
        });
    }

    // ═══════════════════════════════════════
    // CAPTURE & UPLOAD
    // ═══════════════════════════════════════
    if(captureBtn) captureBtn.addEventListener('click', () => captureAndProcess());

    const triggerUploadBtn = document.getElementById('triggerUploadBtn');
    if(triggerUploadBtn && photoUpload) {
        triggerUploadBtn.addEventListener('click', () => photoUpload.click());
    }

    if(photoUpload) photoUpload.addEventListener('change', e => {
        const file = e.target.files[0]; if(!file) return;
        const reader = new FileReader();
        reader.onload = async ev => {
            const base64 = ev.target.result;
            // Stop the camera view but don't show the static image
            stopPipMirror();
            if(videoElement){videoElement.pause();videoElement.style.opacity='0';}
            if(bracketOverlay) bracketOverlay.style.display = 'none';
            await captureAndProcess(base64);
        };
        reader.readAsDataURL(file);
    });

    async function captureAndProcess(providedBase64=null) {
        if(isCapturing) return;
        isCapturing = true;
        if(captureBtn){captureBtn.disabled=true; captureBtn.textContent='Processing...';}

        let base64 = providedBase64;
        if(!base64 && cameraReady && videoElement.videoWidth>0) {
            const maxDim = 600;
            let w = videoElement.videoWidth;
            let h = videoElement.videoHeight;
            if (w > maxDim || h > maxDim) {
                const scale = maxDim / Math.max(w, h);
                w = Math.floor(w * scale);
                h = Math.floor(h * scale);
            }
            captureCanvas.width = w;
            captureCanvas.height = h;
            captureCanvas.getContext('2d').drawImage(videoElement,0,0,w,h);
            base64 = captureCanvas.toDataURL('image/jpeg',0.6);
        }

        // Show processing UI
        if(processingCard) processingCard.style.display='flex';
        if(resultPreviewCard) resultPreviewCard.style.display='none';

        if(base64) {
            try {
                if(processingStatus) processingStatus.textContent = 'Validating image...';
                
                const valRes = await fetch(`${API_BASE}/api/validate`,{
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({image_base64:base64, foot_side:currentFoot})
                });
                const valData = await valRes.json();
                
                if(!valData.valid) {
                    if(processingStatus) processingStatus.textContent = valData.message || 'Validation failed. Adjust and retry.';
                    setTimeout(()=>{
                        if(processingCard) processingCard.style.display='none';
                        isCapturing = false;
                        if(captureBtn){captureBtn.disabled=false; captureBtn.innerHTML='<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg> CAPTURE';}
                        if(videoElement && cameraReady){videoElement.style.opacity='1'; videoElement.play().catch(()=>{});}
                        startPipMirror();
                    }, 3000);
                    return;
                }

                animateProcessing();

                const res = await fetch(`${API_BASE}/api/measure`,{
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({image_base64:base64, foot_side:currentFoot})
                });
                const data = await res.json();
                if(res.ok && data.length_cm) {
                    const uk  = data.shoe_size_uk;
                    const us  = data.shoe_size_us;
                    const eu  = Math.round(data.length_cm*1.5+2);
                    const ind = uk;
                    const result = {foot:currentFoot, length_cm:data.length_cm, width_cm:data.width_cm, uk, us, eu, ind};
                    await saveResult(result);
                    showQuickResult(result);
                    if(processingCard) processingCard.style.display='none';
                    if(resultPreviewCard) resultPreviewCard.style.display='block';
                } else {
                    if(processingStatus) processingStatus.textContent = data.message||data.detail||'Detection failed. Adjust and retry.';
                    setTimeout(()=>{if(processingCard)processingCard.style.display='none';},3000);
                }
            } catch(e) {
                if(processingStatus) processingStatus.textContent = 'Server unreachable. Check backend.';
                setTimeout(()=>{if(processingCard)processingCard.style.display='none';},3000);
            }
        } else {
            if(processingStatus) processingStatus.textContent = 'No image captured. Try Gallery.';
            setTimeout(()=>{if(processingCard)processingCard.style.display='none';},2500);
        }

        isCapturing = false;
        if(captureBtn){captureBtn.disabled=false; captureBtn.innerHTML='<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg> CAPTURE';}
        if(photoUpload) photoUpload.value='';
        // Restore camera
        const li = document.getElementById('loadedPhotoPreview');
        if(li){li.style.display='none';}
        if(bracketOverlay) bracketOverlay.style.display = 'block';
        if(videoElement && cameraReady){videoElement.style.opacity='1'; videoElement.play().catch(()=>{});}
        startPipMirror();
    }

    function animateProcessing() {
        if(!pStepsSm.length) return;
        const steps = ['detect','segment','measure'];
        pStepsSm.forEach(s=>s.classList.remove('active','done'));
        steps.forEach((step,i)=>{
            const el = document.querySelector(`.p-step-sm[data-step="${step}"]`);
            setTimeout(()=>{
                pStepsSm.forEach(s=>s.classList.remove('active'));
                if(el) el.classList.add('active');
                if(processingStatus) processingStatus.textContent = ['Detecting A4 paper...','Segmenting foot...','Calculating sizes...'][i];
            }, i*1200);
            setTimeout(()=>{if(el){el.classList.remove('active');el.classList.add('done');}},i*1200+1100);
        });
    }

    function showQuickResult(r) {
        const el = document.getElementById('quickResult');
        if(!el) return;
        el.innerHTML = `
            <div class="qr-size">${r.ind}</div>
            <div style="font-size:12px;color:rgba(255,255,255,.5);margin-bottom:8px;">${r.foot.toUpperCase()} FOOT · ${r.length_cm} cm × ${r.width_cm} cm</div>
            <div class="qr-row">
                <span class="qr-chip">🇮🇳 IND ${r.ind}</span>
                <span class="qr-chip">🇺🇸 US ${r.us}</span>
                <span class="qr-chip">🇬🇧 UK ${r.uk}</span>
                <span class="qr-chip">🇪🇺 EU ${r.eu}</span>
            </div>`;
        refreshDashboard();
    }

});
