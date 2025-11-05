import streamlit as st
import numpy as np
from copy import deepcopy
from collections import deque

# Game constants
EMPTY = 0
BLUE = 1
RED = 2

COLOR_MAP = {
    EMPTY: "⬜",
    BLUE: "🔵",
    RED: "🔴"
}

PLAYER_NAMES = {
    BLUE: "Blue",
    RED: "Red"
}

class ExpandConquerGame:
    def __init__(self, rows, cols, variant="normal", starting_player=BLUE):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.variant = variant
        self.current_player = starting_player
        self.starting_player = starting_player
        self.move_history = []
        
    def set_piece(self, row, col, color):
        """Set a piece on the board during setup"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.board[row, col] = color
            
    def get_connected_group(self, row, col):
        """Get all cells in the connected group containing (row, col)"""
        if self.board[row, col] == EMPTY:
            return set()
        
        color = self.board[row, col]
        visited = set()
        queue = deque([(row, col)])
        group = set()
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                continue
            if self.board[r, c] != color:
                continue
                
            visited.add((r, c))
            group.add((r, c))
            
            # Add orthogonally adjacent cells
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))
        
        return group
    
    def get_expansion_cells(self, group):
        """Get all cells that would be filled by expanding a group"""
        expansion = set()
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.board[nr, nc] == EMPTY:
                        expansion.add((nr, nc))
        return expansion
    
    def get_all_groups(self, color):
        """Get all connected groups for a color"""
        visited = set()
        groups = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in visited and self.board[r, c] == color:
                    group = self.get_connected_group(r, c)
                    if group:
                        groups.append(group)
                        visited.update(group)
        
        return groups
    
    def get_isolated_squares(self, color):
        """Get all squares not orthogonally adjacent to any group (for all-small variant)"""
        if self.variant != "all-small":
            return []
        
        # Get all cells adjacent to existing groups
        adjacent_cells = set()
        groups = self.get_all_groups(color)
        for group in groups:
            for r, c in group:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        adjacent_cells.add((nr, nc))
        
        # Find empty cells not adjacent to any group
        isolated = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == EMPTY and (r, c) not in adjacent_cells:
                    isolated.append((r, c))
        
        return isolated
    
    def get_legal_moves(self, color):
        """Get all legal moves for a color"""
        moves = []
        
        # Normal expansion moves
        groups = self.get_all_groups(color)
        for i, group in enumerate(groups):
            expansion = self.get_expansion_cells(group)
            if expansion:
                # Use first cell of group as representative
                rep_cell = min(group)
                moves.append({
                    'type': 'expand',
                    'group_id': i,
                    'representative': rep_cell,
                    'group': group,
                    'expansion': expansion
                })
        
        # All-small variant: isolated square placement
        if self.variant == "all-small":
            isolated = self.get_isolated_squares(color)
            for cell in isolated:
                moves.append({
                    'type': 'place',
                    'cell': cell
                })
        
        return moves
    
    def make_move(self, move):
        """Execute a move"""
        color = self.current_player
        
        if move['type'] == 'expand':
            for r, c in move['expansion']:
                self.board[r, c] = color
        elif move['type'] == 'place':
            r, c = move['cell']
            self.board[r, c] = color
        
        self.move_history.append({
            'player': color,
            'move': move,
            'board_state': deepcopy(self.board)
        })
        
        # Switch player
        self.current_player = RED if self.current_player == BLUE else BLUE
    
    def has_moves(self, color):
        """Check if a color has any legal moves"""
        return len(self.get_legal_moves(color)) > 0
    
    def is_game_over(self):
        """Check if game is over"""
        return not self.has_moves(self.current_player)
    
    def get_winner(self):
        """Get the winner (the player who just moved, since current player has no moves)"""
        if self.is_game_over():
            return RED if self.current_player == BLUE else BLUE
        return None

def render_interactive_board(game, hovered_move=None):
    """Render the game board with interactive cells showing move previews"""
    
    # Get all legal moves for current player
    legal_moves = game.get_legal_moves(game.current_player)
    
    # Create a mapping of cells to their moves
    cell_to_move = {}
    for move in legal_moves:
        if move['type'] == 'expand':
            # Map each cell in the group to this move
            for r, c in move['group']:
                if (r, c) not in cell_to_move:
                    cell_to_move[(r, c)] = move
        elif move['type'] == 'place':
            r, c = move['cell']
            cell_to_move[(r, c)] = move
    
    # Determine which cells to highlight
    highlight_cells = set()
    highlight_type = None
    if hovered_move:
        if hovered_move['type'] == 'expand':
            highlight_cells = hovered_move['expansion']
            highlight_type = 'expansion'
        elif hovered_move['type'] == 'place':
            highlight_cells = {hovered_move['cell']}
            highlight_type = 'place'
    
    # Create HTML table with clickable cells
    cell_size = 60 if max(game.rows, game.cols) <= 6 else 50
    
    html = f'<div style="display: flex; justify-content: center; margin: 20px;">'
    html += f'<table style="border-collapse: collapse; border: 3px solid #333;">'
    
    for r in range(game.rows):
        html += '<tr>'
        for c in range(game.cols):
            color = game.board[r, c]
            emoji = COLOR_MAP[color]
            
            # Determine cell styling
            is_highlighted = (r, c) in highlight_cells
            is_clickable = (r, c) in cell_to_move
            
            # Base styling
            bg_color = '#ffffff'
            border = '1px solid #999'
            cursor = 'default'
            
            if is_highlighted:
                if highlight_type == 'expansion':
                    bg_color = '#fff59d'  # Yellow for expansion
                elif highlight_type == 'place':
                    bg_color = '#81c784'  # Green for placement
            elif is_clickable and color != EMPTY:
                bg_color = '#e3f2fd'  # Light blue for clickable groups
                cursor = 'pointer'
            elif is_clickable and color == EMPTY:
                bg_color = '#c8e6c9'  # Light green for isolated placement
                cursor = 'pointer'
            
            html += f'<td id="cell_{r}_{c}" '
            html += f'style="width: {cell_size}px; height: {cell_size}px; '
            html += f'text-align: center; border: {border}; '
            html += f'background-color: {bg_color}; font-size: 30px; '
            html += f'cursor: {cursor}; user-select: none;"'
            
            if is_clickable:
                html += f' onmouseenter="window.parent.postMessage({{type: \'hover\', row: {r}, col: {c}}}, \'*\')" '
                html += f' onmouseleave="window.parent.postMessage({{type: \'unhover\'}}, \'*\')" '
                html += f' onclick="window.parent.postMessage({{type: \'click\', row: {r}, col: {c}}}, \'*\')"'
            
            html += f'>{emoji}</td>'
        html += '</tr>'
    html += '</table></div>'
    
    return html, cell_to_move

def render_board_simple(game):
    """Simple non-interactive board rendering"""
    cell_size = 60 if max(game.rows, game.cols) <= 6 else 50
    
    html = f'<div style="display: flex; justify-content: center; margin: 20px;">'
    html += f'<table style="border-collapse: collapse; border: 3px solid #333;">'
    
    for r in range(game.rows):
        html += '<tr>'
        for c in range(game.cols):
            color = game.board[r, c]
            emoji = COLOR_MAP[color]
            
            html += f'<td style="width: {cell_size}px; height: {cell_size}px; '
            html += f'text-align: center; border: 1px solid #999; '
            html += f'background-color: #ffffff; font-size: 30px;">{emoji}</td>'
        html += '</tr>'
    html += '</table></div>'
    
    return html

def main():
    st.set_page_config(page_title="Expand and Conquer", layout="wide")
    
    st.title("🎮 Expand and Conquer")
    st.markdown("*A combinatorial game by Aditya Khambete*")
    
    # Initialize session state
    if 'game' not in st.session_state:
        st.session_state.game = None
        st.session_state.setup_mode = True
        st.session_state.hovered_move = None
        st.session_state.cell_to_move = {}
    
    # Sidebar for game configuration
    with st.sidebar:
        st.header("⚙️ Game Configuration")
        
        # Variant selection
        variant = st.radio("Variant", ["normal", "all-small"], 
                          help="All-small allows placing pieces in isolated squares")
        
        # Board size
        st.subheader("Board Size")
        size_preset = st.radio("Preset", ["Custom", "1×n", "n×1", "Square"])
        
        if size_preset == "1×n":
            cols = st.slider("Length (n)", 2, 15, 6)
            rows = 1
        elif size_preset == "n×1":
            rows = st.slider("Length (n)", 2, 15, 6)
            cols = 1
        elif size_preset == "Square":
            size = st.slider("Size", 2, 10, 4)
            rows = cols = size
        else:  # Custom
            cols1, cols2 = st.columns(2)
            with cols1:
                rows = st.number_input("Rows", 1, 10, 4)
            with cols2:
                cols = st.number_input("Cols", 1, 15, 4)
        
        # Starting player selection
        st.subheader("Starting Player")
        starting_player_name = st.radio("Who moves first?", ["Blue", "Red"],
                                       help="Fundamental for CGT analysis!")
        starting_player = BLUE if starting_player_name == "Blue" else RED
        
        # New game button
        if st.button("🆕 New Game", type="primary", use_container_width=True):
            st.session_state.game = ExpandConquerGame(rows, cols, variant, starting_player)
            st.session_state.setup_mode = True
            st.session_state.hovered_move = None
            st.session_state.cell_to_move = {}
            st.rerun()
        
        # Game rules
        with st.expander("📖 Game Rules"):
            st.markdown("""
            **Normal Variant:**
            - Click a group to expand it to adjacent empty squares
            - Last player to move wins
            
            **All-Small Variant:**
            - Same as normal, PLUS
            - Can place piece on isolated empty squares
            - Ensures both players always have moves
            
            **Controls:**
            - Setup: Click cells to place/remove pieces
            - Play: Click your groups to expand them
            - Hover to preview moves
            """)
    
    # Main game area
    if st.session_state.game is None:
        st.info("👈 Configure and start a new game from the sidebar")
        return
    
    game = st.session_state.game
    
    # Setup mode
    if st.session_state.setup_mode:
        st.subheader("🎨 Setup Mode")
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            setup_action = st.radio("Action", ["Place Blue", "Place Red", "Clear Cell"],
                                   horizontal=True)
        with col4:
            if st.button("▶️ Start Playing", type="primary"):
                st.session_state.setup_mode = False
                st.rerun()
        
        st.markdown("**Click cells on the board to set up your position**")
        
        # Render board for setup
        board_html = render_board_simple(game)
        st.markdown(board_html, unsafe_allow_html=True)
        
        # Setup controls via form
        st.markdown("---")
        st.markdown("**Or use coordinates:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            setup_row = st.number_input("Row", 0, game.rows - 1, 0, key="setup_row")
        with col2:
            setup_col = st.number_input("Col", 0, game.cols - 1, 0, key="setup_col")
        with col3:
            if st.button("Place Blue"):
                game.set_piece(setup_row, setup_col, BLUE)
                st.rerun()
        with col4:
            if st.button("Place Red"):
                game.set_piece(setup_row, setup_col, RED)
                st.rerun()
        
        if st.button("Clear Board"):
            game.board = np.zeros((game.rows, game.cols), dtype=int)
            st.rerun()
    
    # Play mode
    else:
        # Check game over
        if game.is_game_over():
            winner = game.get_winner()
            winner_name = PLAYER_NAMES[winner]
            st.success(f"🎉 Game Over! **{winner_name}** wins!")
            st.balloons()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Restart with same position"):
                    # Reset to starting position
                    game.current_player = game.starting_player
                    game.move_history = []
                    # Undo all moves
                    if len(game.move_history) > 0:
                        game.board = game.move_history[0]['board_state']
                    st.rerun()
            with col2:
                if st.button("🎨 Back to Setup"):
                    st.session_state.setup_mode = True
                    st.rerun()
        
        # Current player info
        current_name = PLAYER_NAMES[game.current_player]
        emoji = COLOR_MAP[game.current_player]
        st.subheader(f"{emoji} **{current_name}'s Turn** (Move {len(game.move_history) + 1})")
        
        # Render interactive board
        board_html, cell_to_move = render_interactive_board(game, st.session_state.hovered_move)
        st.session_state.cell_to_move = cell_to_move
        
        # Instructions
        legal_moves = game.get_legal_moves(game.current_player)
        if legal_moves:
            expand_moves = sum(1 for m in legal_moves if m['type'] == 'expand')
            place_moves = sum(1 for m in legal_moves if m['type'] == 'place')
            
            if expand_moves > 0 and place_moves > 0:
                st.info(f"💡 {expand_moves} group(s) to expand (light blue) | {place_moves} isolated square(s) to place (light green)")
            elif expand_moves > 0:
                st.info(f"💡 Click on any of your {expand_moves} group(s) (light blue) to expand")
            else:
                st.info(f"💡 Click on any of {place_moves} isolated square(s) (light green) to place a piece")
        
        st.markdown(board_html, unsafe_allow_html=True)
        
        # Move buttons (as fallback or for 1xn boards where clicking might be hard)
        if legal_moves:
            with st.expander("📋 List of Available Moves (click here if board clicking doesn't work)"):
                cols = st.columns(min(4, len(legal_moves)))
                for idx, move in enumerate(legal_moves):
                    with cols[idx % 4]:
                        if move['type'] == 'expand':
                            rep_r, rep_c = move['representative']
                            btn_label = f"Expand at ({rep_r},{rep_c})"
                        else:
                            r, c = move['cell']
                            btn_label = f"Place at ({r},{c})"
                        
                        if st.button(btn_label, key=f"move_{idx}", use_container_width=True):
                            game.make_move(move)
                            st.session_state.hovered_move = None
                            st.rerun()
        
        # Controls
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("↩️ Undo Move") and len(game.move_history) > 0:
                # Restore previous state
                prev_state = game.move_history[-1]
                game.move_history.pop()
                if len(game.move_history) > 0:
                    game.board = game.move_history[-1]['board_state'].copy()
                else:
                    # Go back to initial state - need to track this
                    game.board = np.zeros((game.rows, game.cols), dtype=int)
                game.current_player = RED if game.current_player == BLUE else BLUE
                st.rerun()
        with col2:
            if st.button("🎨 Back to Setup"):
                st.session_state.setup_mode = True
                st.rerun()
        with col3:
            if st.button("🔄 Restart Game"):
                game.current_player = game.starting_player
                game.move_history = []
                st.rerun()
        
        # Move history
        if len(game.move_history) > 0:
            with st.expander(f"📜 Move History ({len(game.move_history)} moves)"):
                for idx, move_data in enumerate(game.move_history):
                    player_name = PLAYER_NAMES[move_data['player']]
                    emoji = COLOR_MAP[move_data['player']]
                    move = move_data['move']
                    
                    if move['type'] == 'expand':
                        desc = f"Expanded group (+{len(move['expansion'])} cells)"
                    else:
                        r, c = move['cell']
                        desc = f"Placed at ({r},{c})"
                    
                    st.text(f"{idx+1}. {emoji} {player_name}: {desc}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built for IE619 project | Supports Normal and All-Small variants | Perfect for CGT analysis*")

if __name__ == "__main__":
    main()
