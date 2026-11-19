// Real-time Voice Agent Dashboard Client Logic
(function () {
  let ws = null;
  let reconnectTimer = null;
  let targetRms = 0.0;
  let currentRms = 0.0;
  let turnCount = 0;
  let startTime = Date.now();

  // Elements
  const wsIndicator = document.getElementById('ws-indicator');
  const wsStatusText = document.getElementById('ws-status-text');
  const uptimeDisplay = document.getElementById('uptime-display');
  const turnStatePill = document.getElementById('turn-state-pill');
  const partialTranscript = document.getElementById('partial-transcript');
  const micLevelLabel = document.getElementById('mic-level-label');
  const vadSlider = document.getElementById('vad-slider');
  const vadVal = document.getElementById('vad-val');
  const speechRateSlider = document.getElementById('speech-rate-slider');
  const speechRateVal = document.getElementById('speech-rate-val');
  const bargeinToggle = document.getElementById('bargein-toggle');
  const timelineContainer = document.getElementById('timeline-container');
  const timelineEmpty = document.getElementById('timeline-empty');
  const turnCounter = document.getElementById('turn-counter');
  const metricPerceived = document.getElementById('metric-perceived');
  const metricTtft = document.getElementById('metric-ttft');
  const metricStt = document.getElementById('metric-stt');
  const metricTts = document.getElementById('metric-tts');
  const barStt = document.getElementById('bar-stt');
  const barLlm = document.getElementById('bar-llm');
  const barTts = document.getElementById('bar-tts');
  const valStt = document.getElementById('val-stt');
  const valLlm = document.getElementById('val-llm');
  const valTts = document.getElementById('val-tts');
  const refreshBtn = document.getElementById('refresh-metrics-btn');
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas ? canvas.getContext('2d') : null;

  // Uptime ticker
  setInterval(() => {
    const sec = Math.floor((Date.now() - startTime) / 1000);
    if (uptimeDisplay) uptimeDisplay.textContent = `Uptime: ${sec}s`;
  }, 1000);

  // Canvas waveform visualizer animation loop
  let phase = 0;
  function drawWaveform() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Smoothly interpolate current RMS toward target
    currentRms += (targetRms - currentRms) * 0.15;
    targetRms *= 0.95; // Decay

    const width = canvas.width;
    const height = canvas.height;
    const midY = height / 2;

    // Draw grid background line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // Draw dynamic sine waveform
    const baseAmp = 4 + currentRms * 180;
    phase += 0.08 + currentRms * 0.15;

    ctx.beginPath();
    ctx.lineWidth = 2.5;
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, 'rgba(0, 229, 255, 0.2)');
    gradient.addColorStop(0.5, 'rgba(0, 229, 255, 0.9)');
    gradient.addColorStop(1, 'rgba(157, 78, 221, 0.8)');
    ctx.strokeStyle = gradient;

    for (let x = 0; x < width; x++) {
      const norm = (x / width) * Math.PI * 4;
      const envelope = Math.sin((x / width) * Math.PI); // Window tapering at edges
      const y = midY + Math.sin(norm + phase) * baseAmp * envelope + Math.cos(norm * 2 - phase) * (baseAmp * 0.4) * envelope;
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    requestAnimationFrame(drawWaveform);
  }
  requestAnimationFrame(drawWaveform);

  // WebSocket Connection
  function connectWs() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const wsUrl = `${protocol}//${host}/ws/events`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      if (wsIndicator) wsIndicator.className = 'status-indicator online';
      if (wsStatusText) wsStatusText.textContent = 'Live Connected';
      console.log('[WS] Connected to Voice Agent Gateway');
    };

    ws.onclose = () => {
      if (wsIndicator) wsIndicator.className = 'status-indicator';
      if (wsStatusText) wsStatusText.textContent = 'Disconnected';
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectWs, 2500);
    };

    ws.onerror = () => {
      if (ws) ws.close();
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleWsEvent(payload);
      } catch (err) {
        console.error('WS parse error:', err);
      }
    };
  }

  function handleWsEvent(msg) {
    const type = msg.type;
    const data = msg.data || {};

    if (type === 'audio_level') {
      targetRms = data.rms || 0.0;
      if (micLevelLabel) micLevelLabel.textContent = `Mic RMS: ${targetRms.toFixed(3)}`;
      const noiseFloorLabel = document.getElementById('noise-floor-label');
      const snrLabel = document.getElementById('snr-label');
      if (noiseFloorLabel && data.noise_floor_rms !== undefined) {
        const dbfs = (20 * Math.log10(Math.max(data.noise_floor_rms, 1e-6))).toFixed(1);
        noiseFloorLabel.textContent = `Floor: ${dbfs} dBFS`;
      }
      if (snrLabel && data.snr_db !== undefined) {
        snrLabel.textContent = `SNR: ${data.snr_db.toFixed(1)} dB`;
      }
    } else if (type === 'state_change') {
      updateTurnState(data.state || 'IDLE');
    } else if (type === 'partial_transcript') {
      if (partialTranscript) {
        partialTranscript.textContent = data.text || '...';
        partialTranscript.style.color = '#00e5ff';
      }
    } else if (type === 'final_transcript') {
      if (partialTranscript) {
        partialTranscript.textContent = `[Finalized]: "${data.text}"`;
        partialTranscript.style.color = '#f0f4fc';
      }
      appendTimelineEntry('user', data.text, data.turn_id);
    } else if (type === 'agent_filler') {
      appendTimelineFiller(data.text);
    } else if (type === 'agent_response') {
      appendTimelineEntry('agent', data.text, data.turn_id);
    } else if (type === 'barge_in_detected') {
      updateTurnState('INTERRUPTED');
      if (partialTranscript) {
        partialTranscript.textContent = '⚡ Barge-in interruption triggered!';
        partialTranscript.style.color = '#ef4444';
      }
    } else if (type === 'turn_completed') {
      updateMetrics(data);
      turnCount++;
      if (turnCounter) turnCounter.textContent = `Turns: ${turnCount}`;
    } else if (type === 'persona_changed') {
      if (personaSelect && data.name) {
        personaSelect.value = data.name;
      }
      if (speechRateSlider && data.speech_rate !== undefined) {
        speechRateSlider.value = data.speech_rate;
        if (speechRateVal) speechRateVal.textContent = `${Number(data.speech_rate).toFixed(2)}x`;
      }
    } else if (type === 'tts_config_updated') {
      if (speechRateSlider && data.speech_rate !== undefined) {
        speechRateSlider.value = data.speech_rate;
        if (speechRateVal) speechRateVal.textContent = `${Number(data.speech_rate).toFixed(2)}x`;
      }
    } else if (type === 'connection_established') {
      if (speechRateSlider && data.speech_rate !== undefined) {
        speechRateSlider.value = data.speech_rate;
        if (speechRateVal) speechRateVal.textContent = `${Number(data.speech_rate).toFixed(2)}x`;
      }
    }
  }

  function updateTurnState(state) {
    if (!turnStatePill) return;
    turnStatePill.textContent = state;
    turnStatePill.className = `turn-state-pill ${state}`;
  }

  function appendTimelineEntry(role, text, turnId) {
    if (timelineEmpty) timelineEmpty.style.display = 'none';
    const entry = document.createElement('div');
    entry.className = `timeline-entry ${role}`;
    const timeStr = new Date().toLocaleTimeString();

    entry.innerHTML = `
      <div class="timeline-meta">
        <span class="timeline-role">${role.toUpperCase()} ${turnId ? `(Turn #${turnId})` : ''}</span>
        <span>${timeStr}</span>
      </div>
      <p class="timeline-text">${escapeHtml(text)}</p>
    `;
    timelineContainer.appendChild(entry);
    timelineContainer.scrollTop = timelineContainer.scrollHeight;
  }

  function appendTimelineFiller(fillerText) {
    const fillerEl = document.createElement('div');
    fillerEl.className = 'timeline-filler';
    fillerEl.textContent = `⤷ Predictive Filler: "${fillerText}"`;
    timelineContainer.appendChild(fillerEl);
    timelineContainer.scrollTop = timelineContainer.scrollHeight;
  }

  function updateMetrics(summary) {
    if (!summary) return;
    if (metricPerceived && summary.perceived_latency_ms !== null) {
      metricPerceived.textContent = `${summary.perceived_latency_ms}ms`;
    }
    if (metricTtft && summary.llm_ttft_ms !== null) {
      metricTtft.textContent = `${summary.llm_ttft_ms}ms`;
    }
    if (metricStt && summary.stt_ms !== null) {
      metricStt.textContent = `${summary.stt_ms}ms`;
    }
    if (metricTts && summary.tts_first_audio_ms !== null) {
      metricTts.textContent = `${summary.tts_first_audio_ms}ms`;
    }

    // Waterfall bars
    const stt = summary.stt_ms || 20;
    const llm = summary.llm_ttft_ms || 50;
    const tts = summary.tts_first_audio_ms || 30;
    const total = stt + llm + tts;

    if (valStt) valStt.textContent = `${stt}ms`;
    if (valLlm) valLlm.textContent = `${llm}ms`;
    if (valTts) valTts.textContent = `${tts}ms`;

    if (barStt) barStt.style.width = `${Math.min(100, (stt / total) * 100)}%`;
    if (barLlm) barLlm.style.width = `${Math.min(100, (llm / total) * 100)}%`;
    if (barTts) barTts.style.width = `${Math.min(100, (tts / total) * 100)}%`;
  }

  async function fetchMetrics() {
    try {
      const resp = await fetch('/api/metrics');
      if (resp.ok) {
        const json = await resp.json();
        if (json.turn_history && json.turn_history.length > 0) {
          updateMetrics(json.turn_history[json.turn_history.length - 1]);
        }
      }
    } catch (e) {
      console.warn('Metrics fetch error', e);
    }
  }

  // Parameter controls
  if (vadSlider) {
    vadSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (vadVal) vadVal.textContent = val.toFixed(2);
    });
    vadSlider.addEventListener('change', async (e) => {
      const val = parseFloat(e.target.value);
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vad_threshold: val }),
      });
    });
  }

  if (bargeinToggle) {
    bargeinToggle.addEventListener('change', (e) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'toggle_barge_in' }));
      }
    });
  }

  if (refreshBtn) {
    refreshBtn.addEventListener('click', fetchMetrics);
  }

  // Persona management
  const personaSelect = document.getElementById('persona-select');
  async function loadPersonas() {
    try {
      const resp = await fetch('/api/personas');
      if (resp.ok) {
        const data = await resp.json();
        if (personaSelect && data.personas) {
          personaSelect.innerHTML = '';
          data.personas.forEach((p) => {
            const opt = document.createElement('option');
            opt.value = p.name;
            opt.textContent = `${p.display_name}`;
            if (p.is_active) opt.selected = true;
            personaSelect.appendChild(opt);
          });
        }
      }
    } catch (e) {
      console.warn('Personas fetch error', e);
    }
  }

  if (personaSelect) {
    personaSelect.addEventListener('change', async (e) => {
      const name = e.target.value;
      const resp = await fetch('/api/personas/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (resp.ok) {
        const res = await resp.json();
        if (res.speech_rate !== undefined && speechRateSlider) {
          speechRateSlider.value = res.speech_rate;
          if (speechRateVal) speechRateVal.textContent = `${Number(res.speech_rate).toFixed(2)}x`;
        }
      }
    });
  }

  // TTS Prosody / Speech Rate Control
  async function loadTtsConfig() {
    try {
      const resp = await fetch('/api/tts/config');
      if (resp.ok) {
        const data = await resp.json();
        if (speechRateSlider && data.speech_rate !== undefined) {
          speechRateSlider.value = data.speech_rate;
          if (speechRateVal) speechRateVal.textContent = `${Number(data.speech_rate).toFixed(2)}x`;
        }
      }
    } catch (e) {
      console.warn('TTS config fetch error', e);
    }
  }

  if (speechRateSlider) {
    speechRateSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (speechRateVal) speechRateVal.textContent = `${val.toFixed(2)}x`;
    });
    speechRateSlider.addEventListener('change', async (e) => {
      const val = parseFloat(e.target.value);
      await fetch('/api/tts/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speech_rate: val }),
      });
    });
  }

  // Web Audio Browser Voice Call (Microphone Ingest & Speaker Output)
  let voiceWs = null;
  let audioContext = null;
  let micMediaStream = null;
  let micSourceNode = null;
  let processorNode = null;
  let activeAudioSource = null;
  let isCallActive = false;

  const btnVoiceCall = document.getElementById('btn-voice-call');
  const callBtnText = document.getElementById('call-btn-text');
  const browserMicStatus = document.getElementById('browser-mic-status');

  async function startBrowserVoiceCall() {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      micMediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host || 'localhost:8000';
      voiceWs = new WebSocket(`${protocol}//${host}/ws/voice`);
      voiceWs.binaryType = 'arraybuffer';

      voiceWs.onopen = () => {
        isCallActive = true;
        if (btnVoiceCall) btnVoiceCall.className = 'btn-voice-call active';
        if (callBtnText) callBtnText.textContent = 'End Voice Call';
        if (browserMicStatus) {
          browserMicStatus.className = 'browser-mic-badge active';
          browserMicStatus.textContent = 'Mic Live';
        }

        // Setup audio worklet / ScriptProcessor for 16kHz float32 frames
        micSourceNode = audioContext.createMediaStreamSource(micMediaStream);
        processorNode = audioContext.createScriptProcessor(512, 1, 1);

        processorNode.onaudioprocess = (e) => {
          if (!isCallActive || !voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;
          const inputData = e.inputBuffer.getChannelData(0);
          // Send Float32Array bytes to backend
          voiceWs.send(inputData.buffer);

          // Update local waveform
          let sum = 0;
          for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
          const rms = Math.sqrt(sum / inputData.length);
          targetRms = Math.max(targetRms, rms);
          if (micLevelLabel) micLevelLabel.textContent = `Mic RMS: ${rms.toFixed(3)}`;
        };

        micSourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
      };

      voiceWs.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Received synthesized audio PCM from backend for speaker playback
          playAgentAudio(event.data);
        } else {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'barge_in') {
              stopAgentAudio();
              updateTurnState('INTERRUPTED');
            } else if (data.type === 'transcript') {
              appendTimelineEntry('user', data.text);
            } else if (data.type === 'agent_response') {
              appendTimelineEntry('agent', data.text);
            }
          } catch (err) {}
        }
      };

      voiceWs.onclose = () => {
        stopBrowserVoiceCall();
      };
    } catch (err) {
      console.error('Error starting browser mic call:', err);
      alert('Could not access microphone: ' + err.message);
      stopBrowserVoiceCall();
    }
  }

  let audioQueue = [];
  let isPlayingAudio = false;

  function playAgentAudio(arrayBuffer) {
    if (!audioContext) return;
    try {
      const floatArr = new Float32Array(arrayBuffer);
      const audioBuffer = audioContext.createBuffer(1, floatArr.length, 22050);
      audioBuffer.copyToChannel(floatArr, 0);
      audioQueue.push(audioBuffer);
      if (!isPlayingAudio) {
        playNextInQueue();
      }
    } catch (e) {
      console.warn('Audio queue error', e);
    }
  }

  function playNextInQueue() {
    if (audioQueue.length === 0) {
      isPlayingAudio = false;
      updateTurnState('IDLE');
      return;
    }
    isPlayingAudio = true;
    updateTurnState('AGENT_SPEAKING');
    const buf = audioQueue.shift();
    activeAudioSource = audioContext.createBufferSource();
    activeAudioSource.buffer = buf;
    activeAudioSource.connect(audioContext.destination);
    activeAudioSource.onended = () => {
      activeAudioSource = null;
      playNextInQueue();
    };
    activeAudioSource.start();
  }

  function stopAgentAudio() {
    audioQueue = [];
    isPlayingAudio = false;
    if (activeAudioSource) {
      try {
        activeAudioSource.stop();
      } catch (e) {}
      activeAudioSource = null;
    }
  }

  function stopBrowserVoiceCall() {
    isCallActive = false;
    stopAgentAudio();
    if (processorNode) {
      processorNode.disconnect();
      processorNode = null;
    }
    if (micSourceNode) {
      micSourceNode.disconnect();
      micSourceNode = null;
    }
    if (micMediaStream) {
      micMediaStream.getTracks().forEach((t) => t.stop());
      micMediaStream = null;
    }
    if (voiceWs) {
      if (voiceWs.readyState === WebSocket.OPEN) voiceWs.close();
      voiceWs = null;
    }
    if (btnVoiceCall) btnVoiceCall.className = 'btn-voice-call';
    if (callBtnText) callBtnText.textContent = 'Start Browser Voice Call';
    if (browserMicStatus) {
      browserMicStatus.className = 'browser-mic-badge';
      browserMicStatus.textContent = 'Offline';
    }
  }

  if (btnVoiceCall) {
    btnVoiceCall.addEventListener('click', () => {
      if (!isCallActive) {
        startBrowserVoiceCall();
      } else {
        stopBrowserVoiceCall();
      }
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
    });
  }

  // Session History & Audio Replay
  const sessionSelect = document.getElementById('session-select');
  const sessionsContainer = document.getElementById('sessions-container');
  const refreshSessionsBtn = document.getElementById('refresh-sessions-btn');
  const exportSessionBtn = document.getElementById('export-session-btn');

  async function loadSessions() {
    try {
      const res = await fetch('/api/sessions');
      if (!res.ok) return;
      const data = await res.json();
      if (!sessionSelect) return;
      sessionSelect.innerHTML = '';
      if (!data.sessions || data.sessions.length === 0) {
        sessionSelect.innerHTML = '<option value="">No recorded sessions</option>';
        if (sessionsContainer) sessionsContainer.innerHTML = '<div class="timeline-empty">No recorded sessions yet.</div>';
        return;
      }
      data.sessions.forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s.session_id;
        opt.textContent = `${s.session_id} (${s.total_turns} turns)`;
        sessionSelect.appendChild(opt);
      });
      loadSessionDetail(data.sessions[0].session_id);
    } catch (e) {
      console.warn('Could not load sessions', e);
    }
  }

  async function loadSessionDetail(sessionId) {
    if (!sessionId) return;
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
      if (!res.ok) return;
      const session = await res.json();
      if (!sessionsContainer) return;
      sessionsContainer.innerHTML = '';
      if (!session.turns || session.turns.length === 0) {
        sessionsContainer.innerHTML = '<div class="timeline-empty">Session has no turns recorded yet.</div>';
        return;
      }
      session.turns.forEach((t) => {
        const card = document.createElement('div');
        card.className = 'session-turn-card';
        const perceived = t.metrics && t.metrics.perceived_latency_ms ? `${t.metrics.perceived_latency_ms}ms` : '--';
        card.innerHTML = `
          <div class="session-turn-header">
            <span>Turn #${t.turn_id} &bull; Persona: ${escapeHtml(t.persona || 'assistant')}</span>
            <span>Perceived Latency: <strong style="color:var(--accent-cyan);">${perceived}</strong></span>
          </div>
          <div><strong style="color:var(--accent-cyan);">User:</strong> ${escapeHtml(t.user_text || '(audio only)')}</div>
          <div><strong style="color:var(--accent-purple);">Agent:</strong> ${escapeHtml(t.agent_text || '')}</div>
          <div class="audio-controls-row">
            ${t.user_audio_file ? `<div class="audio-player-pill"><span>User Audio:</span><audio controls src="/api/sessions/${encodeURIComponent(sessionId)}/audio/${t.turn_id}/user"></audio></div>` : ''}
            ${t.agent_audio_file ? `<div class="audio-player-pill"><span>Agent Audio:</span><audio controls src="/api/sessions/${encodeURIComponent(sessionId)}/audio/${t.turn_id}/agent"></audio></div>` : ''}
          </div>
        `;
        sessionsContainer.appendChild(card);
      });
    } catch (e) {
      console.warn('Error loading session details', e);
    }
  }

  if (sessionSelect) {
    sessionSelect.addEventListener('change', () => loadSessionDetail(sessionSelect.value));
  }
  if (refreshSessionsBtn) {
    refreshSessionsBtn.addEventListener('click', () => loadSessions());
  }
  if (exportSessionBtn) {
    exportSessionBtn.addEventListener('click', async () => {
      const sid = sessionSelect ? sessionSelect.value : '';
      if (!sid) return alert('Select a session to export');
      const res = await fetch(`/api/sessions/${encodeURIComponent(sid)}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: 'markdown' }),
      });
      if (res.ok) {
        const text = await res.text();
        const blob = new Blob([text], { type: 'text/markdown' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${sid}.md`;
        a.click();
      }
    });
  }

  const btnCalibrateMic = document.getElementById('btn-calibrate-mic');
  const agcToggle = document.getElementById('agc-toggle');

  if (btnCalibrateMic) {
    btnCalibrateMic.addEventListener('click', async () => {
      btnCalibrateMic.textContent = '⏳ Calibrating...';
      try {
        const res = await fetch('/api/audio/calibrate', { method: 'POST' });
        const data = await res.json();
        if (data.profile) {
          const p = data.profile;
          btnCalibrateMic.textContent = `✅ Calibrated (${p.noise_floor_dbfs} dBFS)`;
          if (vadSlider && p.recommended_threshold) {
            vadSlider.value = p.recommended_threshold;
            if (vadVal) vadVal.textContent = Number(p.recommended_threshold).toFixed(2);
          }
          setTimeout(() => {
            btnCalibrateMic.textContent = '🎯 Calibrate Ambient Noise';
          }, 3000);
        }
      } catch (e) {
        btnCalibrateMic.textContent = '❌ Calibration Failed';
      }
    });
  }

  if (agcToggle) {
    agcToggle.addEventListener('change', async () => {
      await fetch('/api/audio/agc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: agcToggle.checked }),
      });
    });
  }

  // Local Knowledge Base (Voice RAG)
  const ragToggle = document.getElementById('rag-toggle');
  const kbTitle = document.getElementById('kb-title');
  const kbContent = document.getElementById('kb-content');
  const kbAddBtn = document.getElementById('kb-add-btn');
  const kbQuery = document.getElementById('kb-query');
  const kbQueryBtn = document.getElementById('kb-query-btn');
  const kbResults = document.getElementById('kb-results');

  async function loadKnowledge() {
    try {
      const res = await fetch('/api/knowledge');
      if (!res.ok) return;
      const data = await res.json();
      if (ragToggle) ragToggle.checked = Boolean(data.rag_enabled);
      if (!kbResults) return;
      if (!data.documents || data.documents.length === 0) {
        kbResults.innerHTML = '<div class="timeline-empty">No documents indexed yet.</div>';
        return;
      }
      kbResults.innerHTML = '';
      data.documents.forEach((doc) => {
        const row = document.createElement('div');
        row.className = 'kb-item';
        row.innerHTML = `
          <span><strong>${escapeHtml(doc.title)}</strong> (${doc.chunk_count} chunks)</span>
          <button class="btn-refresh" data-doc-id="${escapeHtml(doc.doc_id)}">Delete</button>
        `;
        row.querySelector('button').addEventListener('click', async () => {
          await fetch(`/api/knowledge/documents/${encodeURIComponent(doc.doc_id)}`, { method: 'DELETE' });
          loadKnowledge();
        });
        kbResults.appendChild(row);
      });
    } catch (e) {
      console.warn('Could not load knowledge base', e);
    }
  }

  if (kbAddBtn) {
    kbAddBtn.addEventListener('click', async () => {
      const title = kbTitle ? kbTitle.value.trim() : '';
      const content = kbContent ? kbContent.value.trim() : '';
      if (!title || !content) return;
      const res = await fetch('/api/knowledge/documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, content }),
      });
      if (res.ok) {
        if (kbTitle) kbTitle.value = '';
        if (kbContent) kbContent.value = '';
        loadKnowledge();
      }
    });
  }

  if (kbQueryBtn) {
    kbQueryBtn.addEventListener('click', async () => {
      const query = kbQuery ? kbQuery.value.trim() : '';
      if (!query) return loadKnowledge();
      const res = await fetch('/api/knowledge/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 3 }),
      });
      if (!res.ok || !kbResults) return;
      const data = await res.json();
      if (!data.results || data.results.length === 0) {
        kbResults.innerHTML = '<div class="timeline-empty">No matching snippets found.</div>';
        return;
      }
      kbResults.innerHTML = '';
      data.results.forEach((hit) => {
        const row = document.createElement('div');
        row.className = 'kb-item';
        row.innerHTML = `<span><strong>${escapeHtml(hit.title)}</strong> (score ${hit.score}): ${escapeHtml(hit.content)}</span>`;
        kbResults.appendChild(row);
      });
    });
  }

  if (ragToggle) {
    ragToggle.addEventListener('change', async () => {
      await fetch('/api/knowledge/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: ragToggle.checked }),
      });
    });
  }

  // Initialize
  connectWs();
  fetchMetrics();
  loadPersonas();
  loadTtsConfig();
  loadSessions();
  loadKnowledge();
})();
