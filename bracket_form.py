import matplotlib.pyplot as plt
import numpy as np
import warnings

# Constants
ORTHOGONAL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # No diagonals

# Original game functions
def parse_board(board_str):
    """Convert board string to a 2D list representation and validate legality."""
    rows = board_str.split("|")
    row_lengths = {len(row) for row in rows}
    
    if len(row_lengths) > 1:
        raise ValueError("Invalid board: Rows must all be the same length.")
        
    allowed_chars = {"O", "X", "*"}
    for row in rows:
        if not set(row).issubset(allowed_chars):
            raise ValueError("Invalid board: Only 'O', 'X', and '*' are allowed.")
            
    return [list(row) for row in rows]
    
def get_groups(board, player):
    """Find all groups (connected components) of the given player."""
    groups = []
    visited = set()
    rows, cols = len(board), len(board[0])

    def dfs(r, c, group):
        if (r, c) in visited or board[r][c] != player:
            return
        visited.add((r, c))
        group.append((r, c))
        for dr, dc in ORTHOGONAL_DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc, group)

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == player and (r, c) not in visited:
                group = []
                dfs(r, c, group)
                groups.append(group)
    
    return groups

def get_valid_moves(board, group):
    """Get all empty spaces orthogonally adjacent to a group."""
    rows, cols = len(board), len(board[0])
    moves = set()

    for r, c in group:
        for dr, dc in ORTHOGONAL_DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == "*":
                moves.add((nr, nc))
    
    return list(moves)

def get_isolated_squares(board, player):
    """Find all isolated empty squares where the player can place a piece."""
    rows, cols = len(board), len(board[0])
    isolated = []
    for r in range(rows):
        for c in range(cols):
            if board[r][c] != "*":
                continue  # Not an empty square

            has_adjacent = any(
                0 <= r+dr < rows and 0 <= c+dc < cols and board[r+dr][c+dc] == player
                for dr, dc in ORTHOGONAL_DIRECTIONS
            )
            if not has_adjacent:  # If no adjacent pieces, it's isolated
                isolated.append((r, c))

    return isolated

# Bracket form computation functions
def compute_bracket_form(board, use_allsmall=False, max_depth=None, current_depth=0, memo=None):
    """
    Compute the bracket form for a given board position.
    
    Args:
        board: 2D list representing the current board state
        use_allsmall: Whether to use the AllSmall ruleset
        max_depth: Maximum recursion depth (None for unlimited)
        current_depth: Current recursion depth
        memo: Dictionary for memoization
    """
    if memo is None:
        memo = {}
    
    # Convert board to a string for memoization
    board_key = board_to_key(board)
    
    if board_key in memo:
        return memo[board_key]
    
    # Check depth limit
    if max_depth is not None and current_depth >= max_depth:
        warnings.warn(f"Reached maximum depth of {max_depth}. Bracket form is incomplete.")
        return "{DEPTH_LIMIT}"
    
    # Get Left player (O) options
    left_options = []
    for new_board in get_player_moves(board, "O", use_allsmall):
        option_value = compute_bracket_form(new_board, use_allsmall, max_depth, 
                                            current_depth + 1, memo)
        left_options.append(option_value)
    
    # Get Right player (X) options
    right_options = []
    for new_board in get_player_moves(board, "X", use_allsmall):
        option_value = compute_bracket_form(new_board, use_allsmall, max_depth, 
                                            current_depth + 1, memo)
        right_options.append(option_value)
    
    # Remove duplicates (same positions)
    left_options = list(set(left_options))
    right_options = list(set(right_options))
    
    # Build the bracket notation
    bracket_form = "{" + "|".join([",".join(left_options), ",".join(right_options)]) + "}"
    
    # Apply standard CGT simplifications for common values
    if not left_options and not right_options:
        bracket_form = "0"  # Game over, no moves for either player - value 0
    
    # Save result in memo
    memo[board_key] = bracket_form
    
    return bracket_form

def get_player_moves(board, player, use_allsmall=False):
    """Get all possible resulting board states after the player makes a move."""
    all_boards = []
    
    # Get all groups for the player
    groups = get_groups(board, player)
    
    # For each group, create a new board state after expansion
    for group in groups:
        moves = get_valid_moves(board, group)
        if moves:
            # Create a copy of the board
            new_board = [row[:] for row in board]
            
            # Apply the expansion move (fill ALL adjacent empty spaces)
            for r, c in moves:
                new_board[r][c] = player
                
            all_boards.append(new_board)
    
    # If using AllSmall ruleset, also consider isolated placements
    if use_allsmall:
        isolated_squares = get_isolated_squares(board, player)
        for r, c in isolated_squares:
            new_board = [row[:] for row in board]
            new_board[r][c] = player
            all_boards.append(new_board)
    
    return all_boards

def board_to_key(board):
    """Convert a board to a string key for memoization."""
    return '|'.join([''.join(row) for row in board])

def format_bracket_form(bracket_form):
    """Format the bracket form for display, ensuring proper syntax."""
    # Handle special cases
    if bracket_form in ["0", "{DEPTH_LIMIT}"]:
        return bracket_form
    
    # Remove outer brackets
    if bracket_form.startswith("{") and bracket_form.endswith("}"):
        content = bracket_form[1:-1]
    else:
        return bracket_form  # Return as is if not in bracket form
    
    # Split at the pipe character
    parts = content.split("|")
    if len(parts) != 2:
        return bracket_form  # Return as is if not properly formatted
    
    left_part, right_part = parts
    
    # Process left options
    left_options = []
    if left_part:
        left_options = [opt.strip() for opt in left_part.split(",") if opt.strip()]
    
    # Process right options
    right_options = []
    if right_part:
        right_options = [opt.strip() for opt in right_part.split(",") if opt.strip()]
    
    # Recreate the bracket form with proper formatting
    formatted = "{" + "|".join([",".join(left_options), ",".join(right_options)]) + "}"
    
    return formatted

def visualize_board_array(board):
    """Visualize a 2D board array using matplotlib."""
    rows, cols = len(board), len(board[0])
    color_map = {"O": "orange", "X": "green", "*": "white"}
    
    fig, ax = plt.subplots(figsize=(cols * 0.5, rows * 0.5))
    
    # Setup grid
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(which="both", color="black", linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which="both", bottom=False, left=False)
    
    # Draw grid squares
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, edgecolor="black", facecolor="white"))
    
    # Draw pieces
    for r in range(rows):
        for c in range(cols):
            if board[r][c] in color_map and board[r][c] != "*":
                ax.add_patch(plt.Circle((c + 0.5, rows - r - 0.5), 0.3, 
                                        color=color_map[board[r][c]], ec="black"))
    plt.show()

def analyze_game_position(board_str, use_allsmall=False, max_depth=None):
    """Analyze a game position and display its bracket form."""
    # Parse the board
    board = parse_board(board_str)
    
    print("Board visualization:")
    visualize_board_array(board)
    
    print(f"Computing bracket form for {'AllSmall' if use_allsmall else 'Normal'} ruleset...")
    if max_depth:
        print(f"Using depth limit: {max_depth}")
    else:
        print("No depth limit (may take a long time for complex positions)")
        
    # Compute the bracket form
    bracket_form = compute_bracket_form(board, use_allsmall, max_depth)
    
    # Format for consistency
    formatted = format_bracket_form(bracket_form)
    
    return formatted

# Example usage
if __name__ == "__main__":
    # Example 2x2 board
    board_str = "O*|*X"
    
    print("Normal Ruleset Bracket Form:")
    result = analyze_game_position(board_str, use_allsmall=False, max_depth=5)
    print(result)
    
    print("\nAllSmall Ruleset Bracket Form:")
    result = analyze_game_position(board_str, use_allsmall=True, max_depth=5)
    print(result)
