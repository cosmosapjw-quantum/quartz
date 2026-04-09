const stateStore = {
  models: [],
  session: null,
  selectedSquare: null,
  thinking: false,
};

const els = {
  gameSelect: document.getElementById("game-select"),
  modelSelect: document.getElementById("model-select"),
  sideSelect: document.getElementById("side-select"),
  itersInput: document.getElementById("iters-input"),
  newGameBtn: document.getElementById("new-game-btn"),
  undoBtn: document.getElementById("undo-btn"),
  restartBtn: document.getElementById("restart-btn"),
  passBtn: document.getElementById("pass-btn"),
  aiBtn: document.getElementById("ai-btn"),
  resignBtn: document.getElementById("resign-btn"),
  statusText: document.getElementById("status-text"),
  sessionMeta: document.getElementById("session-meta"),
  turnBadge: document.getElementById("turn-badge"),
  gameMeta: document.getElementById("game-meta"),
  thinking: document.getElementById("thinking-indicator"),
  board: document.getElementById("board"),
  history: document.getElementById("history-list"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function groupedModels(game) {
  return stateStore.models.filter((entry) => entry.game === game);
}

function refreshModelSelect() {
  const game = els.gameSelect.value;
  const items = groupedModels(game);
  els.modelSelect.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = item.label;
    els.modelSelect.appendChild(option);
  }
}

async function loadModels() {
  const data = await api("/api/models");
  stateStore.models = data.models;
  els.gameSelect.innerHTML = "";
  for (const game of data.games) {
    const option = document.createElement("option");
    option.value = game;
    option.textContent = game;
    els.gameSelect.appendChild(option);
  }
  if (data.games.length > 0) {
    const firstPlayable = data.games.find((game) => groupedModels(game).length > 0) || data.games[0];
    els.gameSelect.value = firstPlayable;
  }
  refreshModelSelect();
}

function currentSession() {
  return stateStore.session;
}

async function createSession() {
  const payload = {
    game: els.gameSelect.value,
    modelPath: els.modelSelect.value,
    humanSide: els.sideSelect.value,
    searchIterations: Number(els.itersInput.value || 1),
  };
  const session = await api("/api/session", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  stateStore.session = session;
  stateStore.selectedSquare = null;
  renderSession();
  maybeRunAi();
}

async function refreshSession() {
  const session = currentSession();
  if (!session) return;
  stateStore.session = await api(`/api/session/${session.sessionId}`);
  renderSession();
}

async function sessionAction(action, payload = {}) {
  const session = currentSession();
  if (!session) return;
  stateStore.session = await api(`/api/session/${session.sessionId}/${action}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (action !== "ai") {
    stateStore.selectedSquare = null;
  }
  renderSession();
  maybeRunAi();
}

function renderEmptyBoard() {
  els.board.className = "board empty";
  els.board.textContent = "Start a session to play.";
}

function renderHistory(session) {
  els.history.innerHTML = "";
  for (const move of session.moveHistory) {
    const item = document.createElement("li");
    item.textContent = `${move.ply}. ${move.side} ${move.label}`;
    els.history.appendChild(item);
  }
}

function stoneClass(game, value) {
  if (game === "tictactoe") return "";
  return value === 1 ? "black" : "white";
}

function renderGridBoard(session) {
  els.board.className = "board grid-board";
  els.board.style.setProperty("--size", String(session.boardSize));
  els.board.innerHTML = "";
  const legal = new Set(session.legalActions || []);
  for (let i = 0; i < session.board.length; i += 1) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "cell";
    if (session.humanToMove && legal.has(i)) {
      cell.classList.add("playable");
      cell.addEventListener("click", () => sessionAction("move", { action: i }));
    } else {
      cell.disabled = true;
    }

    const value = session.board[i];
    if (session.game === "tictactoe" && value !== 0) {
      const mark = document.createElement("div");
      mark.className = "ttt-mark";
      mark.textContent = value === 1 ? "X" : "O";
      cell.appendChild(mark);
    } else if (value !== 0) {
      const stone = document.createElement("div");
      stone.className = `stone ${stoneClass(session.game, value)} ${session.game.startsWith("gomoku") ? "gomoku" : ""}`;
      cell.appendChild(stone);
    }
    els.board.appendChild(cell);
  }
}

function chessPieceGlyph(piece) {
  const map = {
    K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
    k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
  };
  return map[piece] || "";
}

function legalMovesFrom(square, session) {
  return (session.legalMoves || []).filter((move) => move.from === square);
}

function handleChessSquareClick(index) {
  const session = currentSession();
  if (!session || !session.humanToMove) return;
  const fromMoves = legalMovesFrom(index, session);
  if (stateStore.selectedSquare === null) {
    if (fromMoves.length > 0) {
      stateStore.selectedSquare = index;
      renderSession();
    }
    return;
  }
  if (stateStore.selectedSquare === index) {
    stateStore.selectedSquare = null;
    renderSession();
    return;
  }
  const candidates = (session.legalMoves || []).filter(
    (move) => move.from === stateStore.selectedSquare && move.to === index
  );
  if (candidates.length === 0) {
    if (fromMoves.length > 0) {
      stateStore.selectedSquare = index;
      renderSession();
    } else {
      stateStore.selectedSquare = null;
      renderSession();
    }
    return;
  }
  if (candidates.length === 1) {
    sessionAction("move", { move_uci: candidates[0].uci });
    return;
  }
  const promotion = window.prompt("Promotion piece? Enter q, r, b, or n", "q");
  const chosen = candidates.find((move) => move.promotion === (promotion || "q").toLowerCase());
  if (chosen) {
    sessionAction("move", { move_uci: chosen.uci });
  }
}

function renderChessBoard(session) {
  els.board.className = "board chess-board";
  els.board.innerHTML = "";
  const selected = stateStore.selectedSquare;
  const targets = new Set();
  if (selected !== null) {
    for (const move of legalMovesFrom(selected, session)) {
      targets.add(move.to);
    }
  }
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const i = row * 8 + col;
      const squareIndex = (7 - row) * 8 + col;
    const square = document.createElement("button");
    square.type = "button";
    square.className = "square";
    square.classList.add((row + col) % 2 === 0 ? "light" : "dark");
    if (session.humanToMove) {
      square.classList.add("playable");
      square.addEventListener("click", () => handleChessSquareClick(squareIndex));
    } else {
      square.disabled = true;
    }
      if (selected === squareIndex) square.classList.add("selected");
      if (targets.has(squareIndex)) square.classList.add("legal-target");
      square.textContent = chessPieceGlyph(session.board[squareIndex]);
    els.board.appendChild(square);
  }
  }
}

function renderSession() {
  const session = currentSession();
  if (!session) {
    renderEmptyBoard();
    els.history.innerHTML = "";
    els.turnBadge.textContent = "Idle";
    els.gameMeta.textContent = "";
    els.sessionMeta.textContent = "No active session.";
    return;
  }

  setStatus(session.resultLabel);
  els.turnBadge.textContent = session.terminal ? session.resultLabel : `${session.currentPlayer} to move`;
  const modelName = session.modelPath.split("/").pop();
  let meta = `${session.game} | ${modelName} | ${session.searchIterations} iters`;
  if (session.ruleset) {
    meta += ` | ${session.ruleset} ${session.scoring}`;
  }
  els.gameMeta.textContent = meta;
  els.sessionMeta.textContent = `${session.humanSide} vs ${session.aiSide}`;

  if (session.render === "chess") {
    renderChessBoard(session);
  } else {
    renderGridBoard(session);
  }
  renderHistory(session);
  els.passBtn.disabled = !(session.humanToMove && session.passAction !== null);
  els.aiBtn.disabled = !session.aiToMove;
  els.undoBtn.disabled = session.moveHistory.length === 0;
  els.restartBtn.disabled = false;
  els.resignBtn.disabled = session.terminal;
}

async function maybeRunAi() {
  const session = currentSession();
  if (!session || !session.aiToMove || session.terminal || stateStore.thinking) return;
  stateStore.thinking = true;
  els.thinking.classList.remove("hidden");
  try {
    await sessionAction("ai");
  } catch (err) {
    setStatus(err.message);
  } finally {
    stateStore.thinking = false;
    els.thinking.classList.add("hidden");
  }
}

function bindEvents() {
  els.gameSelect.addEventListener("change", refreshModelSelect);
  els.newGameBtn.addEventListener("click", async () => {
    try {
      await createSession();
    } catch (err) {
      setStatus(err.message);
    }
  });
  els.undoBtn.addEventListener("click", () => sessionAction("undo", { count: 2 }));
  els.restartBtn.addEventListener("click", () => sessionAction("restart", {}));
  els.passBtn.addEventListener("click", () => {
    const session = currentSession();
    if (session && session.passAction !== null) {
      sessionAction("move", { action: session.passAction });
    }
  });
  els.aiBtn.addEventListener("click", () => sessionAction("ai", {}));
  els.resignBtn.addEventListener("click", () => {
    const session = currentSession();
    if (session) {
      sessionAction("resign", { side: session.humanSide });
    }
  });
}

async function init() {
  bindEvents();
  renderEmptyBoard();
  try {
    await loadModels();
    setStatus("Choose a game and start.");
  } catch (err) {
    setStatus(err.message);
  }
}

init();
