from food_agent.spatial_store import SpatialContextStore


def test_object_context() -> None:
    store = SpatialContextStore()
    ctx = store.object_context("P01-20240202-110250", time=360.0)
    assert ctx["video_id"] == "P01-20240202-110250"
    assert isinstance(ctx["object_tracks"], list)


def test_combined_context() -> None:
    store = SpatialContextStore()
    ctx = store.combined_context("P01-20240202-110250", time=360.0)
    assert ctx.video_id == "P01-20240202-110250"
    assert isinstance(ctx.object_tracks, list)
    assert isinstance(ctx.audio_events, list)

