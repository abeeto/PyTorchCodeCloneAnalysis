from stem import StreamPurpose, CircStatus
from stem.control import Controller, EventType
from sys import exit


def stream_cb(event):
    print('[circiut_id-stream_id] {}-{}'.format(event.circ_id, event.id))
    print('status is {}, reason: {}'.format(event.status, event.reason))
    print('src is {}'.format(event.source))
    print('dst is {}'.format(event.target))
    print()


def orconn_cb(event):
    print('OR connection event:')
    print('[{}-{}]'.format(event.endpoint_fingerprint, event.endpoint_nickname))
    print('status is {}'.format(event.status))
    print()


def circ_cb(event):
    print('Circuit event:')
    print('status: {}, id'.format(event.status, event.id))
    print('path: {}'.format(event.path))
    print('purpose: {}'.format(event.purpose))
    print('reason: {}, remote reason'.format(event.reason, event.remote_reason))
    print()


if __name__ == '__main__':
    with Controller.from_port(port=9051) as ctrl:
        ctrl.authenticate()

        ctrl.add_event_listener(orconn_cb, EventType.ORCONN)
        ctrl.add_event_listener(circ_cb, EventType.CIRC)

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print('Exiting')
            ctrl.remove_event_listener(orconn_cb)
            ctrl.remove_event_listener(circ_cb)
            exit(1)
