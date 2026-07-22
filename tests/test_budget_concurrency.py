from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from app.db import GatewayStore, ReservationResult


def test_ten_concurrent_vk_edge_reservations_admit_exactly_one(
    test_store: GatewayStore,
) -> None:
    contender_count = 10
    start_together = Barrier(contender_count)

    def reserve() -> ReservationResult:
        start_together.wait()
        return test_store.reserve_request("vk_edge")

    with ThreadPoolExecutor(max_workers=contender_count) as executor:
        results = list(executor.map(lambda _: reserve(), range(contender_count)))

    counts = Counter(results)
    assert counts == Counter({"over_budget": 9, "reserved": 1})

    stats = test_store.get_usage("vk_edge")
    assert stats is not None
    assert stats.requests == 1
    assert stats.budget == 1
    assert stats.as_contract()["remaining"] == 0
