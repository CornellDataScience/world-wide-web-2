"""
data/samples.py

Hardcoded WebInteraction objects for testing with no data files.
These mirror realistic WebArena/NNetNav trajectories.

Store real data at:
    data/raw/train.jsonl
    data/raw/val.jsonl
"""

from data.types import (
    WebInteraction, WebState, WebAction, StateChange, ActionType
)


def make_sample_interactions() -> list:
    """
    Returns a list of 4 hardcoded WebInteraction objects.
    Used automatically when no JSONL file is found.
    """

    # ------------------------------------------------------------------
    # Interaction 1: Search task
    # ------------------------------------------------------------------
    i1_s0 = WebState.from_acc_tree(
        "[button id=1 'Home']\n[textbox id=3 'Search query']\n[button id=4 'Submit']\n[link id=5 'Login']",
        url="https://example.com"
    )
    i1_s1 = WebState.from_acc_tree(
        "[heading id=10 'Search Results']\n[link id=11 'Python Tutorial']\n[link id=12 'Python Docs']\n[button id=14 'Next page']",
        url="https://example.com/search?q=Python+tutorials"
    )
    i1_s2 = WebState.from_acc_tree(
        "[heading id=20 'Python Tutorial - Getting Started']\n[text id=21 'Welcome to Python programming']\n[link id=22 'Chapter 1']",
        url="https://example.com/python-tutorial"
    )
    i1 = WebInteraction(
        task="Search for Python tutorials on the website",
        website="example.com",
        state_changes=[
            StateChange(
                pre_state=i1_s0,
                action=WebAction(ActionType.TYPE, target_id=3, value="Python tutorials", raw="type [3] [Python tutorials] [1]"),
                post_state=i1_s1,
            ),
            StateChange(
                pre_state=i1_s1,
                action=WebAction(ActionType.CLICK, target_id=11, raw="click [11]"),
                post_state=i1_s2,
            ),
            StateChange(
                pre_state=i1_s2,
                action=WebAction(ActionType.STOP, value="Found Python Tutorial page", raw="stop [Found Python Tutorial page]"),
                post_state=i1_s2,
            ),
        ],
        reward=1.0,
        final_state=i1_s2,
        is_real=True,
    )

    # ------------------------------------------------------------------
    # Interaction 2: Login task
    # ------------------------------------------------------------------
    i2_s0 = WebState.from_acc_tree(
        "[button id=1 'Home']\n[link id=5 'Login']\n[link id=6 'Register']\n[textbox id=3 'Search']",
        url="https://example.com"
    )
    i2_s1 = WebState.from_acc_tree(
        "[heading id=30 'Login']\n[textbox id=31 'Username']\n[textbox id=32 'Password']\n[button id=33 'Sign In']",
        url="https://example.com/login"
    )
    i2_s2 = WebState.from_acc_tree(
        "[heading id=40 'Welcome back!']\n[text id=41 'You are logged in as testuser']\n[link id=42 'Dashboard']",
        url="https://example.com/dashboard"
    )
    i2 = WebInteraction(
        task="Log in with username testuser and password password123",
        website="example.com",
        state_changes=[
            StateChange(
                pre_state=i2_s0,
                action=WebAction(ActionType.CLICK, target_id=5, raw="click [5]"),
                post_state=i2_s1,
            ),
            StateChange(
                pre_state=i2_s1,
                action=WebAction(ActionType.TYPE, target_id=31, value="testuser", raw="type [31] [testuser] [0]"),
                post_state=i2_s1,
            ),
            StateChange(
                pre_state=i2_s1,
                action=WebAction(ActionType.TYPE, target_id=32, value="password123", raw="type [32] [password123] [1]"),
                post_state=i2_s2,
            ),
            StateChange(
                pre_state=i2_s2,
                action=WebAction(ActionType.STOP, value="Successfully logged in", raw="stop [Successfully logged in]"),
                post_state=i2_s2,
            ),
        ],
        reward=1.0,
        final_state=i2_s2,
        is_real=True,
    )

    # ------------------------------------------------------------------
    # Interaction 3: Navigation task
    # ------------------------------------------------------------------
    i3_s0 = WebState.from_acc_tree(
        "[button id=1 'Home']\n[link id=7 'Settings']\n[link id=8 'Profile']\n[link id=9 'Help']",
        url="https://example.com"
    )
    i3_s1 = WebState.from_acc_tree(
        "[heading id=50 'Settings']\n[button id=51 'Account Settings']\n[button id=52 'Privacy']\n[button id=53 'Notifications']",
        url="https://example.com/settings"
    )
    i3 = WebInteraction(
        task="Navigate to the settings page",
        website="example.com",
        state_changes=[
            StateChange(
                pre_state=i3_s0,
                action=WebAction(ActionType.CLICK, target_id=7, raw="click [7]"),
                post_state=i3_s1,
            ),
            StateChange(
                pre_state=i3_s1,
                action=WebAction(ActionType.STOP, value="On settings page", raw="stop [On settings page]"),
                post_state=i3_s1,
            ),
        ],
        reward=1.0,
        final_state=i3_s1,
        is_real=True,
    )

    # ------------------------------------------------------------------
    # Interaction 4: Failed task (reward=0)
    # ------------------------------------------------------------------
    i4_s0 = WebState.from_acc_tree(
        "[button id=1 'Home']\n[textbox id=3 'Search']\n[button id=4 'Submit']",
        url="https://example.com"
    )
    i4_s1 = WebState.from_acc_tree(
        "[heading id=60 'No results found']\n[text id=61 'Try a different search term']\n[button id=62 'Search again']",
        url="https://example.com/search?q=xyznotfound"
    )
    i4 = WebInteraction(
        task="Find the price of item XYZ",
        website="example.com",
        state_changes=[
            StateChange(
                pre_state=i4_s0,
                action=WebAction(ActionType.TYPE, target_id=3, value="XYZ price", raw="type [3] [XYZ price] [1]"),
                post_state=i4_s1,
            ),
            StateChange(
                pre_state=i4_s1,
                action=WebAction(ActionType.STOP, value="Could not find item", raw="stop [Could not find item]"),
                post_state=i4_s1,
            ),
        ],
        reward=0.0,
        final_state=i4_s1,
        is_real=True,
    )

    return [i1, i2, i3, i4]