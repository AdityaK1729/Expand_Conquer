"""
Conjecture Testing Script for Expand and Conquer

This script helps test the conjectures from your IE619 report:
- Conjecture 4.2: g_{2n-1} = ↑n 
- Conjecture 5.1: Empty m×n boards are 0 if m*n even, else *

Note: This requires integration with CGSuite or your bracket form computation code
"""

import numpy as np
from expand_conquer_game import ExpandConquerGame
from collections import defaultdict

def analyze_empty_boards(max_size=8):
    """
    Test Conjecture 5.1: Empty boards
    
    Conjecture: For empty boards of size m×n,
    - Game value is 0 if m*n is even
    - Game value is * if m*n is odd
    """
    print("=" * 60)
    print("Testing Conjecture 5.1: Empty Board Values")
    print("=" * 60)
    
    results = {}
    
    for m in range(2, max_size + 1):
        for n in range(2, max_size + 1):
            game = ExpandConquerGame(m, n, variant="all-small")
            
            # Simple heuristic: count who wins with random play
            # (This is NOT computing exact game value, just a quick test)
            blue_wins = 0
            red_wins = 0
            num_trials = 100
            
            for _ in range(num_trials):
                test_game = ExpandConquerGame(m, n, variant="all-small")
                # Simulate random play
                while not test_game.is_game_over():
                    moves = test_game.get_legal_moves(test_game.current_player)
                    if not moves:
                        break
                    move = np.random.choice(moves)
                    test_game.make_move(move)
                
                winner = test_game.get_winner()
                if winner == 1:  # BLUE
                    blue_wins += 1
                elif winner == 2:  # RED
                    red_wins += 1
            
            # Classify based on who wins more often
            if abs(blue_wins - red_wins) < 10:  # Close to 50-50
                estimated_value = "~0 (balanced)"
            elif blue_wins > red_wins:
                estimated_value = ">0 (Blue favored)"
            else:
                estimated_value = "<0 (Red favored)"
            
            predicted = "0" if (m * n) % 2 == 0 else "*"
            
            results[(m, n)] = {
                'estimated': estimated_value,
                'predicted': predicted,
                'blue_wins': blue_wins,
                'red_wins': red_wins
            }
            
            match = "✓" if (estimated_value == "~0 (balanced)" and predicted == "0") else "?"
            print(f"{m}×{n}: m*n={'even' if (m*n)%2==0 else 'odd':4s} | "
                  f"Predicted: {predicted:1s} | "
                  f"Blue:{blue_wins:3d} Red:{red_wins:3d} | {match}")
    
    print("\n" + "=" * 60)
    print("Note: This uses random play simulation, not exact solving.")
    print("For rigorous testing, integrate with CGSuite or bracket form code.")
    print("=" * 60)
    
    return results

def analyze_1xn_positions(max_n=7):
    """
    Test properties of 1×n board positions
    
    From your report:
    - X * * ... * O gives specific values
    - Patterns like X * O * O give dyadic rationals
    """
    print("\n" + "=" * 60)
    print("Analyzing 1×n Board Positions")
    print("=" * 60)
    
    # Test integer values (Theorem 3.2)
    print("\nTheorem 3.2: Blue in corner with n blank squares = game value n")
    for n in range(1, max_n + 1):
        game = ExpandConquerGame(1, n + 1, variant="normal")
        game.set_piece(0, 0, 1)  # Blue in corner
        # Remaining squares are empty
        
        moves_blue = len(game.get_legal_moves(1))
        moves_red = len(game.get_legal_moves(2))
        
        print(f"  n={n}: Board={n+1} cells | Blue moves: {moves_blue} | Red moves: {moves_red}")
        print(f"    Expected value: {n}")
    
    # Test dyadic rationals (Theorem 3.5)
    print("\nTheorem 3.5: X * * ... * O * O gives 1/2^k")
    test_cases = [
        ("X O * O", 1, "1"),           # 0 stars between X and first O
        ("X * O * O", 1, "1/2"),       # 1 star
        ("X * * O * O", 2, "1/4"),     # 2 stars
        ("X * * * O * O", 3, "1/8"),   # 3 stars
    ]
    
    for pattern, k, expected_value in test_cases:
        print(f"\n  Pattern: {pattern}")
        print(f"    k={k} stars between X and O")
        print(f"    Expected value: {expected_value}")
        
        # Setup board
        cells = pattern.split()
        game = ExpandConquerGame(1, len(cells), variant="normal")
        for i, cell in enumerate(cells):
            if cell == 'O':
                game.set_piece(0, i, 1)  # Blue
            elif cell == 'X':
                game.set_piece(0, i, 2)  # Red
        
        moves_blue = len(game.get_legal_moves(1))
        moves_red = len(game.get_legal_moves(2))
        print(f"    Blue moves: {moves_blue} | Red moves: {moves_red}")

def analyze_p_positions(board_size=4):
    """
    Search for P-positions (game value 0) in small boards
    
    A P-position is one where the second player can force a win
    """
    print("\n" + "=" * 60)
    print(f"Searching for P-Positions on {board_size}×{board_size} boards")
    print("=" * 60)
    
    # Test symmetric positions
    print("\nTesting symmetric starting positions:")
    
    # Example 1: Four corners
    game1 = ExpandConquerGame(board_size, board_size, variant="normal")
    game1.set_piece(0, 0, 1)  # Blue
    game1.set_piece(0, board_size-1, 2)  # Red
    game1.set_piece(board_size-1, 0, 2)  # Red
    game1.set_piece(board_size-1, board_size-1, 1)  # Blue
    
    print("\n1. Four corners (Blue: 2, Red: 2):")
    print(f"   Blue moves: {len(game1.get_legal_moves(1))}")
    print(f"   Red moves: {len(game1.get_legal_moves(2))}")
    
    # Example 2: Center symmetry
    if board_size >= 4:
        game2 = ExpandConquerGame(board_size, board_size, variant="normal")
        mid = board_size // 2
        game2.set_piece(mid-1, mid-1, 1)
        game2.set_piece(mid-1, mid, 2)
        game2.set_piece(mid, mid-1, 2)
        game2.set_piece(mid, mid, 1)
        
        print("\n2. Center 2×2 symmetric:")
        print(f"   Blue moves: {len(game2.get_legal_moves(1))}")
        print(f"   Red moves: {len(game2.get_legal_moves(2))}")
    
    print("\nNote: To confirm P-positions, you need to solve the game tree.")
    print("Use alpha-beta search or integrate with CGSuite.")

def compare_variants():
    """
    Compare game outcomes between Normal and All-Small variants
    """
    print("\n" + "=" * 60)
    print("Comparing Normal vs All-Small Variants")
    print("=" * 60)
    
    # Test case: Single blue piece in corner
    for variant in ["normal", "all-small"]:
        game = ExpandConquerGame(4, 4, variant=variant)
        game.set_piece(0, 0, 1)  # Blue in corner
        
        blue_moves = len(game.get_legal_moves(1))
        red_moves = len(game.get_legal_moves(2))
        
        print(f"\n{variant.upper()} variant:")
        print(f"  Initial position: Blue at (0,0)")
        print(f"  Blue moves available: {blue_moves}")
        print(f"  Red moves available: {red_moves}")
        
        if variant == "normal":
            print(f"  Game value: ~5 (from your report, Fig 17)")
        else:
            print(f"  Game value: ~* (from your report, all-small changes this)")

def generate_position_statistics(num_samples=1000, board_size=4):
    """
    Generate statistics about random positions
    """
    print("\n" + "=" * 60)
    print(f"Position Statistics ({num_samples} random {board_size}×{board_size} boards)")
    print("=" * 60)
    
    outcomes = defaultdict(int)
    move_counts = []
    
    for _ in range(num_samples):
        # Generate random starting position
        game = ExpandConquerGame(board_size, board_size, variant="all-small")
        
        # Randomly place 2-8 pieces
        num_pieces = np.random.randint(2, 9)
        positions = np.random.choice(board_size * board_size, num_pieces, replace=False)
        
        for pos in positions:
            r, c = pos // board_size, pos % board_size
            color = np.random.choice([1, 2])
            game.set_piece(r, c, color)
        
        # Play out randomly
        num_moves = 0
        while not game.is_game_over() and num_moves < 100:
            moves = game.get_legal_moves(game.current_player)
            if not moves:
                break
            move = moves[np.random.randint(len(moves))]
            game.make_move(move)
            num_moves += 1
        
        winner = game.get_winner()
        outcomes[winner] += 1
        move_counts.append(num_moves)
    
    print(f"\nOutcome distribution:")
    print(f"  Blue wins: {outcomes[1]} ({100*outcomes[1]/num_samples:.1f}%)")
    print(f"  Red wins: {outcomes[2]} ({100*outcomes[2]/num_samples:.1f}%)")
    print(f"\nGame length statistics:")
    print(f"  Mean moves: {np.mean(move_counts):.1f}")
    print(f"  Median moves: {np.median(move_counts):.0f}")
    print(f"  Max moves: {np.max(move_counts)}")

def main():
    """Run all analysis functions"""
    print("EXPAND AND CONQUER - CONJECTURE TESTING")
    print("=" * 60)
    print("\nThis script tests conjectures from your IE619 report.")
    print("Note: Random play simulation is used as a proxy for exact solving.")
    print("For rigorous results, integrate with CGSuite or implement alpha-beta.\n")
    
    # Run analyses
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  PART 1: Empty Board Conjecture".ljust(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    analyze_empty_boards(max_size=6)
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  PART 2: 1×n Board Analysis".ljust(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    analyze_1xn_positions(max_n=6)
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  PART 3: P-Position Search".ljust(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    analyze_p_positions(board_size=4)
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  PART 4: Variant Comparison".ljust(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    compare_variants()
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  PART 5: Position Statistics".ljust(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    generate_position_statistics(num_samples=100, board_size=4)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Implement exact game tree solving (alpha-beta)")
    print("2. Integrate with CGSuite for canonical form computation")
    print("3. Test larger board sizes")
    print("4. Verify patterns hold across more examples")
    print("\nFor neural network training: run neural_network_starter.py")
    print("For interactive play: run 'streamlit run expand_conquer_game.py'")

if __name__ == "__main__":
    main()
