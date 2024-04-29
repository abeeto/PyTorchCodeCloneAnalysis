#!/usr/bin/env python3
import sys
import time
from itertools import chain, combinations
from typing import NamedTuple, Sequence, Optional, List, FrozenSet, Set


class People(NamedTuple):
    speed: int


class Side(NamedTuple):
    label: str
    peoples: Sequence[People]


class TurnRecord(NamedTuple):
    from_side_label: str
    to_side_label: str
    peoples: Sequence[People]


class GlobalState:
    def __init__(self):
        self._score = None
        self.start_time = int(time.time())
        self.last_trunc_time = self.start_time
        self.prune_hits = 0

    def get_score(self):
        return self._score

    def set_score(self, value):
        now = int(time.time())

        print(
            'Setting %s, time from start: %s sec, time from last trunc: %s sec' % (
                value,
                now - self.start_time,
                now - self.last_trunc_time,
            ),
            flush=True
        )
        self.last_trunc_time = now

        self._score = value

    score = property(get_score, set_score)


class GameSetup(NamedTuple):
    from_side: Side
    to_side: Side
    history_records: Sequence[TurnRecord]
    start_side_label: str
    target_side_states: Set[FrozenSet[People]] = set()

    @property
    def game_completed(self) -> bool:
        return not self.start_side.peoples

    @property
    def start_side(self) -> Side:
        return [side for side in [self.to_side, self.from_side] if side.label == self.start_side_label][0]

    @property
    def target_side(self) -> Side:
        return self.to_side if self.to_side != self.start_side else self.from_side


def swap_sides(self: GameSetup) -> GameSetup:
    new_setup = GameSetup(*self)._replace(
        from_side=self.to_side,
        to_side=self.from_side,
    )

    return new_setup


def do_transition(setup: GameSetup, peoples_to_transit: Sequence[People]) -> GameSetup:
    record = TurnRecord(setup.from_side.label, setup.to_side.label, peoples_to_transit)

    new_history_records = setup.history_records + [record]

    new_from_side_peoples = tuple(people for people in setup.from_side.peoples if people not in peoples_to_transit)
    new_to_side_peoples = tuple(chain.from_iterable([setup.to_side.peoples, peoples_to_transit]))

    new_setup = GameSetup(*setup)._replace(
        from_side=Side(*setup.from_side)._replace(peoples=new_from_side_peoples),
        to_side=Side(*setup.to_side)._replace(peoples=new_to_side_peoples),
        history_records=new_history_records,
    )

    if setup.target_side == setup.to_side:
        setup.target_side_states.add(frozenset(setup.target_side.peoples))

    return swap_sides(new_setup)


def turn_score(peoples_to_transit: Sequence[People]) -> int:
    return max([people.speed for people in peoples_to_transit])


def setup_score(setup: GameSetup) -> int:
    return sum(map(lambda rec: turn_score(rec.peoples), setup.history_records))


def trivial_delivery_score(side: Side):
    speeds = [p.speed for p in side.peoples]
    return (len(speeds) - 3) * min(speeds) + sum(speeds)


def prune(setup: GameSetup, score_constraint: int, global_state: GlobalState):
    target = setup.target_side
    if target == setup.from_side and target.peoples:
        if setup_score(setup) > trivial_delivery_score(target):
            return True

        if frozenset(target.peoples) in setup.target_side_states:
            return True

    if score_constraint < 0:
        return True

    if global_state.score and setup_score(setup) > global_state.score:
        return True

    if len(setup.history_records) > 1:
        people_rec1, people_rec2 = map(lambda rec: rec.peoples, setup.history_records[-2:])
        if set(people_rec1) == set(people_rec2):
            return True


def print_prune_stat(global_state):
    print('\r', end='')
    print('Hit prune conditions %s times' % global_state.prune_hits, end='')


def find_completed_with_constraints(setup: GameSetup, score_constraint: int,
                                    global_state: GlobalState) -> List[GameSetup]:

    prune_flag = prune(setup, score_constraint, global_state)
    if prune_flag:
        global_state.prune_hits += 1
        print_prune_stat(global_state)
        return []

    if setup.game_completed:
        score = setup_score(setup)
        if global_state.score is None or score < global_state.score:
            visualize(setup.history_records, make_init_setup())
            global_state.score = score

        return [setup]

    setups = []     # List[GameSetup]

    peoples_to_transit_combinations = [[people] for people in setup.from_side.peoples] + list(
        combinations(setup.from_side.peoples, 2)
    )

    for peoples_to_transit in peoples_to_transit_combinations:
        new_setup = do_transition(setup, peoples_to_transit)
        setups.append(new_setup)

    tails = []
    for setup_ in setups:
        setups_ = find_completed_with_constraints(
            setup_,
            score_constraint - turn_score(setup_.history_records[-1].peoples),
            global_state
        )

        tails.append(setups_)

    return sum(tails, [])


def make_peoples():
    peoples = list(map(People, range(1, 12)))
    # peoples = list(map(People, [1, 101, 102, 1001, 1002, 100001, 100002]))
    # peoples = list(map(People, [1, 2, 3, 4, 5, 6, 7]))

    return peoples


def make_init_setup() -> GameSetup:
    peoples = make_peoples()
    left_side, right_side = Side('left', peoples), Side('right', [])
    history = []
    setup = GameSetup(
        from_side=left_side,
        to_side=right_side,
        history_records=history,
        start_side_label=left_side.label,
    )

    return setup


def main() -> Optional[GameSetup]:
    setup = make_init_setup()
    max_score_bound = trivial_delivery_score(setup.start_side)

    completed_setups = find_completed_with_constraints(setup, max_score_bound, GlobalState())
    if not completed_setups:
        return None

    best_completed_setup = min(completed_setups, key=lambda s: setup_score(s))

    return best_completed_setup


def visualize(history: Sequence[TurnRecord], init_setup: GameSetup, stream=None) -> None:
    arrow = lambda record: (
                               '%s>  ' if record.from_side_label == 'left' else '  <%s'
                           ) % (
            '-- %s --' % ', '.join(map(lambda x: str(x.speed), sorted(record.peoples, key=lambda p: p.speed)))
    )
    join_peoples = lambda side: ', '.join(map(lambda s: str(s), sorted(p.speed for p in side.peoples)))

    lines = []

    current_setup = init_setup
    current_score = 0
    for rec in history:
        score = turn_score(rec.peoples)
        line = 'current score: %s, turn score: %s | {%s} %s {%s}' % (
            current_score,
            score,
            join_peoples(current_setup.start_side),
            arrow(rec),
            join_peoples(current_setup.target_side)
        )

        lines.append(line)
        current_score += score
        current_setup = do_transition(current_setup, rec.peoples)

    lines.append('Total score: %s\n' % current_score)

    stream = stream or sys.stdout
    print('\n\n' + '\n'.join(lines), flush=True, file=stream)


if __name__ == '__main__':
    best_setup = main()
    with open('output.txt', 'w') as f:
        visualize(best_setup.history_records, make_init_setup(), stream=f)
