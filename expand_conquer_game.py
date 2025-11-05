import streamlit as st
import numpy as np
import pandas as pd
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

class ExpandConquerGame:
    def __init__(self, rows, cols, variant="normal"):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.variant = variant
        self.current_player = BLUE
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

def render_board(game, selected_move=None):
    """Render the game board with visualization"""
    board_display = []
    
    # Create a copy for highlighting
    display_board = np.copy(game.board)
    highlight = np.zeros((game.rows, game.cols), dtype=bool)
    
    # Highlight selected move
    if selected_move:
        if selected_move['type'] == 'expand':
            for r, c in selected_move['expansion']:
                highlight[r, c] = True
        elif selected_move['type'] == 'place':
            r, c = selected_move['cell']
            highlight[r, c] = True
    
    # Create HTML table
    html = '<table style="border-collapse: collapse; margin: 20px auto;">'
    for r in range(game.rows):
        html += '<tr>'
        for c in range(game.cols):
            color = display_board[r, c]
            emoji = COLOR_MAP[color]
            
            bg_color = '#ffffcc' if highlight[r, c] else '#ffffff'
            border = '2px solid #000'
            
            html += f'<td style="width: 50px; height: 50px; text-align: center; '
            html += f'border: {border}; background-color: {bg_color}; '
            html += f'font-size: 30px;">{emoji}</td>'
        html += '</tr>'
    html += '</table>'
    
    return html

def main():
    st.set_page_config(page_title="Expand and Conquer", layout="wide")
    
    st.title("🎮 Expand and Conquer")
    st.markdown("*A combinatorial game by Aditya Khambete*")
    
    # Sidebar for game setup
    with st.sidebar:
        st.header("Game Setup")
        
        # Game variant selection
        variant = st.radio("Game Variant", ["normal", "all-small"])
        
        # Board size
        cols1, cols2 = st.columns(2)
        with cols1:
            rows = st.number_input("Rows", min_value=2, max_value=10, value=4)
        with cols2:
            cols = st.number_input("Columns", min_value=2, max_value=10, value=4)
        
        # Initialize or reset game
        if 'game' not in st.session_state or st.button("New Game"):
            st.session_state.game = ExpandConquerGame(rows, cols, variant)
            st.session_state.setup_mode = True
            st.session_state.selected_move = None
            st.rerun()
        
        game = st.session_state.game
        
        # Setup mode controls
        if st.session_state.get('setup_mode', True):
            st.subheader("Setup Board")
            st.write("Click cells below to place pieces")
            
            setup_color = st.radio("Place color:", ["Blue", "Red", "Empty"])
            color_map = {"Blue": BLUE, "Red": RED, "Empty": EMPTY}
            
            # Grid for setup
            st.write("Setup grid:")
            for r in range(game.rows):
                cols_setup = st.columns(game.cols)
                for c in range(game.cols):
                    with cols_setup[c]:
                        current = COLOR_MAP[game.board[r, c]]
                        if st.button(f"{current}", key=f"setup_{r}_{c}"):
                            game.board[r, c] = color_map[setup_color]
                            st.rerun()
            
            if st.button("Start Game", type="primary"):
                st.session_state.setup_mode = False
                st.rerun()
        
        else:
            st.subheader("Game Info")
            current_player_name = "Blue" if game.current_player == BLUE else "Red"
            st.write(f"**Current Player:** {current_player_name}")
            st.write(f"**Move Number:** {len(game.move_history) + 1}")
            
            if st.button("Reset to Setup"):
                st.session_state.setup_mode = True
                st.rerun()
            
            # Show game rules
            with st.expander("Game Rules"):
                st.markdown("""
                **Normal Variant:**
                - Select one of your connected groups
                - Expand it to fill all orthogonally adjacent empty squares
                - Last player to move wins
                
                **All-Small Variant:**
                - Same as normal, PLUS:
                - You can place a piece on any square not adjacent to your groups
                - Ensures both players always have moves until board is full
                """)
    
    # Main game area
    if not st.session_state.get('setup_mode', True):
        game = st.session_state.game
        
        # Check game over
        if game.is_game_over():
            winner = game.get_winner()
            winner_name = "Blue" if winner == BLUE else "Red"
            st.success(f"🎉 Game Over! {winner_name} wins!")
            st.balloons()
        
        # Display board
        st.markdown("### Game Board")
        selected_move = st.session_state.get('selected_move')
        board_html = render_board(game, selected_move)
        st.markdown(board_html, unsafe_allow_html=True)
        
        # Move selection
        if not game.is_game_over():
            st.markdown("### Available Moves")
            
            moves = game.get_legal_moves(game.current_player)
            
            if not moves:
                st.warning("No legal moves available!")
            else:
                # Group expansion moves
                expand_moves = [m for m in moves if m['type'] == 'expand']
                if expand_moves:
                    st.write("**Expand Group:**")
                    cols_moves = st.columns(min(3, len(expand_moves)))
                    for idx, move in enumerate(expand_moves):
                        with cols_moves[idx % 3]:
                            rep_r, rep_c = move['representative']
                            expansion_size = len(move['expansion'])
                            if st.button(f"Group at ({rep_r},{rep_c})\n+{expansion_size} cells", 
                                       key=f"move_{idx}"):
                                game.make_move(move)
                                st.session_state.selected_move = None
                                st.rerun()
                            
                            # Preview button
                            if st.button("👁️ Preview", key=f"preview_{idx}"):
                                st.session_state.selected_move = move
                                st.rerun()
                
                # Place moves (all-small)
                place_moves = [m for m in moves if m['type'] == 'place']
                if place_moves:
                    st.write("**Place Isolated Piece:**")
                    cols_place = st.columns(min(4, len(place_moves)))
                    for idx, move in enumerate(place_moves):
                        with cols_place[idx % 4]:
                            r, c = move['cell']
                            if st.button(f"Place at ({r},{c})", 
                                       key=f"place_{idx}"):
                                game.make_move(move)
                                st.session_state.selected_move = None
                                st.rerun()
                
                if st.button("Clear Preview"):
                    st.session_state.selected_move = None
                    st.rerun()
        
        # Move history
        with st.expander("Move History"):
            for idx, move_data in enumerate(game.move_history):
                player_name = "Blue" if move_data['player'] == BLUE else "Red"
                move = move_data['move']
                if move['type'] == 'expand':
                    st.write(f"{idx+1}. {player_name}: Expanded group (added {len(move['expansion'])} cells)")
                else:
                    r, c = move['cell']
                    st.write(f"{idx+1}. {player_name}: Placed at ({r},{c})")
    
    else:
        # Setup mode display
        st.markdown("### Board Setup")
        st.info("Use the sidebar to place pieces on the board, then click 'Start Game'")
        board_html = render_board(st.session_state.game)
        st.markdown(board_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
