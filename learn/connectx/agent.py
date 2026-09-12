"""ConnectX agent for the Kaggle `connectx` competition.

`my_agent` is deliberately one self-contained function. The course notebook
submits with `inspect.getsource(my_agent)`, so anything the agent needs —
imports, helpers, tables — has to live inside the function body. Nothing is
imported from this repo.

Search: negamax with alpha-beta on a bitboard, iterative deepening under a
wall-clock budget, transposition table, centre-first move ordering. Falls back
to a valid random column if anything at all raises, because an exception in a
ConnectX episode is a loss.
"""


def my_agent(obs, config):
    import random
    import time

    t_start = time.time()

    try:
        COLS = config.columns
        ROWS = config.rows
        K = config.inarow

        board = obs.board
        me = obs.mark
        valid = [c for c in range(COLS) if board[c] == 0]
        if not valid:
            return 0

        # --- bitboard layout -------------------------------------------
        # bit index = col * H + row, row 0 = bottom. H = ROWS + 1 leaves one
        # always-empty sentinel row per column, which is what stops vertical
        # and diagonal runs from wrapping between columns.
        H = ROWS + 1
        DIRS = (1, H, H + 1, H - 1)

        FULL = 0
        BOTTOM = 0
        COLMASK = []
        for c in range(COLS):
            cm = 0
            for r in range(ROWS):
                cm |= 1 << (c * H + r)
            COLMASK.append(cm)
            FULL |= cm
            BOTTOM |= 1 << (c * H)

        my_pos = 0
        mask = 0
        for c in range(COLS):
            for r in range(ROWS):
                cell = board[(ROWS - 1 - r) * COLS + c]
                if cell:
                    b = 1 << (c * H + r)
                    mask |= b
                    if cell == me:
                        my_pos |= b
        op_pos = mask ^ my_pos

        # Centre columns first: they sit in more winning lines, so they produce
        # cutoffs earlier.
        mid = (COLS - 1) / 2.0
        ORDER = sorted(range(COLS), key=lambda c: abs(c - mid))

        # Shift plans for "which empty squares complete a run of K".
        # For each direction and each position of the gap within the run, the
        # other K-1 cells must already be ours.
        PLANS = []
        for d in DIRS:
            for gap in range(K):
                PLANS.append(tuple((i - gap) * d for i in range(K) if i != gap))

        def shift(x, s):
            return x >> s if s >= 0 else x << -s

        def connected(pos):
            for d in DIRS:
                m = pos
                for i in range(1, K):
                    m &= pos >> (d * i)
                    if not m:
                        break
                if m:
                    return True
            return False

        def threat_squares(pos, msk):
            """Empty cells that would complete a K-run for `pos`."""
            res = 0
            for plan in PLANS:
                m = shift(pos, plan[0])
                for s in plan[1:]:
                    m &= shift(pos, s)
                    if not m:
                        break
                if m:
                    res |= m
            return res & FULL & ~msk

        CENTRE = COLMASK[COLS // 2]
        if COLS % 2 == 0:
            CENTRE |= COLMASK[COLS // 2 - 1]

        def evaluate(my, op, msk):
            my_t = threat_squares(my, msk)
            op_t = threat_squares(op, msk)
            playable = (msk + BOTTOM) & FULL
            score = 0
            score += 16 * bin(my_t).count("1") - 16 * bin(op_t).count("1")
            score += 40 * bin(my_t & playable).count("1")
            score -= 40 * bin(op_t & playable).count("1")
            score += 2 * bin(my & CENTRE).count("1")
            score -= 2 * bin(op & CENTRE).count("1")
            return score

        # --- search -----------------------------------------------------
        WIN = 10 ** 6
        INF = 10 ** 9
        budget = 0.90
        counter = [0]
        tt = {}

        class TimeUp(Exception):
            pass

        def negamax(my, op, msk, depth, alpha, beta, ply):
            counter[0] += 1
            if counter[0] & 511 == 0 and time.time() - t_start > budget:
                raise TimeUp

            nb = msk + BOTTOM
            moves = []
            for c in ORDER:
                b = nb & COLMASK[c]
                if b & FULL:
                    if connected(my | b):
                        return WIN - ply
                    moves.append(b)
            if not moves:
                return 0
            if depth == 0:
                return evaluate(my, op, msk)

            key = (my, msk)
            hit = tt.get(key)
            if hit is not None and hit[0] >= depth:
                _, flag, val = hit
                if flag == 0:
                    return val
                if flag == 1 and val > alpha:
                    alpha = val
                elif flag == 2 and val < beta:
                    beta = val
                if alpha >= beta:
                    return val

            alpha0 = alpha
            best = -INF
            for b in moves:
                v = -negamax(op, my | b, msk | b, depth - 1, -beta, -alpha, ply + 1)
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break

            if best <= alpha0:
                flag = 2
            elif best >= beta:
                flag = 1
            else:
                flag = 0
            tt[key] = (depth, flag, best)
            return best

        # --- root -------------------------------------------------------
        nb0 = mask + BOTTOM
        root_moves = []
        for c in ORDER:
            b = nb0 & COLMASK[c]
            if b & FULL:
                root_moves.append((c, b))

        # Win now, and never hand the opponent a win next move.
        for c, b in root_moves:
            if connected(my_pos | b):
                return c
        for c, b in root_moves:
            if connected(op_pos | b):
                return c

        best_move = root_moves[0][0]
        max_depth = ROWS * COLS - bin(mask).count("1")
        depth = 2
        while depth <= max_depth:
            try:
                alpha = -INF
                local_best = None
                for c, b in root_moves:
                    v = -negamax(op_pos, my_pos | b, mask | b, depth - 1, -INF, -alpha, 1)
                    if local_best is None or v > alpha:
                        alpha = v
                        local_best = c
                if local_best is not None:
                    best_move = local_best
                if alpha >= WIN - depth:
                    break
            except TimeUp:
                break
            if time.time() - t_start > budget * 0.5:
                break
            depth += 1

        return int(best_move)

    except Exception:
        try:
            return int(random.choice([c for c in range(config.columns) if obs.board[c] == 0]))
        except Exception:
            return 0
