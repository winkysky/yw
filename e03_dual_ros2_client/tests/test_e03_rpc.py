"""Real loopback Pyro4 transport; synthetic policy, no ROS, model or robot."""

import signal
import threading
import uuid

import numpy as np
import pytest

from test_e03_client import FakePolicy, make_frame, model_config
from clients.e03_protocol import connect_policy


def test_dual_arm_numpy_observations_over_real_pyro_socket():
    Pyro4 = pytest.importorskip("Pyro4")
    import Pyro4.naming
    from shm_transport import expose

    class Service(FakePolicy):
        @expose()
        def get_config(self):
            return model_config()

        @expose()
        def reset(self):
            super().reset()

        @expose()
        def set_prompt(self, prompt):
            super().set_prompt(prompt)

        @expose()
        def add_obs_frame(self, frame):
            super().add_obs_frame(frame)

        @expose()
        def get_action(self, draw_traj=False, compress_traj_img=False):
            return super().get_action(draw_traj, compress_traj_img)

        @Pyro4.expose
        def get_machine_id_via_uuid(self):
            return str(uuid.getnode())

        @Pyro4.expose
        @Pyro4.oneway
        def request_closing_proxy(self, name):
            pass  # No shared memory is used in this socket-only test.

    old_serializers = Pyro4.config.SERIALIZERS_ACCEPTED
    old_timeout = Pyro4.config.COMMTIMEOUT
    old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    Pyro4.config.SERIALIZERS_ACCEPTED = set(old_serializers) | {"pickle"}
    daemon = ns_daemon = policy = None
    workers = []
    try:
        _, ns_daemon, _ = Pyro4.naming.startNS(host="127.0.0.1", port=0, enableBroadcast=False)
        daemon = Pyro4.Daemon(host="127.0.0.1", port=0)
        service = Service()
        uri = daemon.register(service)
        ns_daemon.nameserver.register("e03_dual_test", uri)
        for server in (ns_daemon, daemon):
            worker = threading.Thread(target=server.requestLoop, daemon=True)
            worker.start()
            workers.append(worker)
        policy = connect_policy(dict(uri="e03_dual_test", ns_host="127.0.0.1",
                                     ns_port=ns_daemon.sock.getsockname()[1], timeout_s=2.0),
                                ["external_cam"], 2)
        policy.start("dual arm transport test")
        frame = make_frame(0.0)
        # Force a realistic payload size; socket mode must remain socket mode
        # even when both endpoints happen to be on the same machine.
        frame["external_cam"]["data"]["color"] = np.full((256, 256, 3), 127, np.uint8)
        frame["external_cam"]["camera"].update(width=256, height=256)
        prediction = policy.infer([frame])
        assert prediction.poses.shape == (4, 2, 4, 4)
        np.testing.assert_allclose(prediction.grippers, [[0.2, 0.8]] * 4)
        assert service.frames[0]["external_cam"]["data"]["color"][100, 100, 0] == 127
        assert policy.proxy._force_use_socket is True
    finally:
        if policy is not None:
            policy.proxy._pyro_proxy._pyroRelease()
        for server in (daemon, ns_daemon):
            if server is not None:
                server.shutdown()
                server.close()
        for worker in workers:
            worker.join(timeout=3)
        Pyro4.config.SERIALIZERS_ACCEPTED = old_serializers
        Pyro4.config.COMMTIMEOUT = old_timeout
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
