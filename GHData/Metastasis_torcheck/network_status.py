from stem import Flag
from stem.descriptor.remote import FallbackDirectory
from stem.descriptor import DocumentHandler, parse_file
from stem.descriptor.remote import DescriptorDownloader
from config import CONSENSUS_PATH
from datetime import datetime

HARDCODED_DIRECTORY_IP_LIST = [
    '37.218.247.217',
    '204.13.164.118',
    '199.58.81.140',
    '193.23.244.244',
    '194.109.206.212',
    '86.59.21.38',
    '128.31.0.34',
    '171.25.193.9',
    '154.35.175.225',
    '131.188.40.189'
]


def download_consensus():
    downloader = DescriptorDownloader()
    consensus = downloader.get_consensus(document_handler=DocumentHandler.DOCUMENT).run()[0]

    with open(CONSENSUS_PATH, 'w') as descriptor_file:
        descriptor_file.write(str(consensus))


def get_consensus():
    consensus = None

    try:
        print('Reading cached-consensus')
        consensus = next(parse_file(
            CONSENSUS_PATH,
            document_handler=DocumentHandler.DOCUMENT,
            validate=True
        ))

        if datetime.utcnow() > consensus.valid_until:
            print('Cached consensus is stale')
            consensus = None
    except FileNotFoundError:
        # TODO: add logging here
        print('Consensus not found')
    except ValueError:
        print('Consensus is invalid')
    except Exception:
        pass

    if not consensus:
        print('Downloading consensus...')
        download_consensus()

        try:
            print('Trying read cached-consensus again')
            consensus = next(parse_file(
                CONSENSUS_PATH,
                document_handler=DocumentHandler.DOCUMENT,
                validate=True
            ))
        except FileNotFoundError:
            raise FileNotFoundError
        except ValueError:
            raise ValueError
        except Exception:
            raise Exception

    return consensus


def get_directory_addresses(use_hardcode=False):
    # FIXME: cause we running filtering and tor client on the same machine
    if use_hardcode:
        return HARDCODED_DIRECTORY_IP_LIST

    authorities_ip_list = []
    consensus = get_consensus()

    for fingerprint, relay in consensus.routers.items():
        if Flag.AUTHORITY in relay.flags:
            # print("%s: %s (%s)" % (fingerprint, relay.address, relay.nickname))
            authorities_ip_list.append(relay.address)

    return authorities_ip_list


def get_all_ip():
    ip_list = []
    consensus = get_consensus()

    for fingerprint, relay in consensus.routers.items():
        ip_list.append(relay.address)

    return ip_list


def get_fallbacks():
    try:
        fallback_directories = FallbackDirectory.from_remote()
    except IOError:
        fallback_directories = FallbackDirectory.from_cache()

    fallback_ips = []

    for fallback in fallback_directories.values():
        fallback_ips.append(fallback.address)

    return fallback_ips
