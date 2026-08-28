"""What happens when Home Assistant cannot be reached.

The underlying errors are accurate and useless - "[Errno -3] Temporary failure
in name resolution" describes a DNS lookup perfectly and says nothing about the
.local hostname in a Docker container that caused it. These tests are about the
translation."""

from __future__ import annotations

import httpx
import pytest

from hpmpc.config import HomeAssistantConfig
from hpmpc.ha import HomeAssistant, HomeAssistantError


def client_raising(error: Exception) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ha.local:8123")


def client_returning(status: int) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": "nope"})

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ha.local:8123")


def connect(client: httpx.Client, url: str = "http://homeassistant.local:8123") -> HomeAssistant:
    return HomeAssistant(HomeAssistantConfig(base_url=url, token="t"), client=client)


def test_ping_reports_failure_instead_of_raising():
    ha = connect(client_raising(httpx.ConnectError("[Errno -3] Temporary failure in name resolution")))
    assert ha.ping() is False


def test_a_dns_failure_is_explained_as_dns_not_as_credentials():
    ha = connect(client_raising(httpx.ConnectError("[Errno -3] Temporary failure in name resolution")))
    ha.ping()
    message = ha.diagnose()
    assert "DNS" in message
    assert "not the token" in message
    assert "API key" in message


def test_a_local_hostname_gets_the_docker_explanation():
    """The actual cause, every time someone runs this in a container."""
    ha = connect(client_raising(httpx.ConnectError("[Errno -3] Temporary failure in name resolution")))
    ha.ping()
    message = ha.diagnose()
    assert "mDNS" in message
    assert "192.168" in message, "it should show what to put there instead"


def test_a_plain_hostname_does_not_get_the_mdns_lecture():
    ha = connect(
        client_raising(httpx.ConnectError("[Errno -3] Temporary failure in name resolution")),
        url="http://ha-server:8123",
    )
    ha.ping()
    message = ha.diagnose()
    assert "mDNS" not in message
    assert "ha-server" in message


def test_a_refused_connection_points_at_the_port():
    ha = connect(client_raising(httpx.ConnectError("[Errno 111] Connection refused")))
    ha.ping()
    assert "port" in ha.diagnose()


def test_a_timeout_points_at_the_network_not_the_config():
    ha = connect(client_raising(httpx.ConnectTimeout("timed out")))
    ha.ping()
    assert "firewall" in ha.diagnose()


def test_a_rejected_token_is_named_as_the_token():
    """401 is the one case where a key really is the problem - and it is the
    Home Assistant token, never the key protecting hpmpc's own API."""
    ha = connect(client_returning(401))
    assert ha.ping() is False
    message = ha.diagnose()
    assert "HA_TOKEN" in message
    assert "HPMPC_API_KEY" in message


def test_a_certificate_problem_names_verify_ssl():
    ha = connect(client_raising(httpx.ConnectError("certificate verify failed")))
    ha.ping()
    assert "verify_ssl" in ha.diagnose()


def test_a_successful_ping_leaves_no_error_behind():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": "API running."})

    ha = connect(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ha:8123"))
    assert ha.ping() is True
    assert ha.last_error is None


def test_a_missing_token_says_which_key_is_meant():
    with pytest.raises(HomeAssistantError) as excinfo:
        HomeAssistant(HomeAssistantConfig(base_url="http://ha:8123", token=""))
    assert "HPMPC_API_KEY is a different thing" in str(excinfo.value)
