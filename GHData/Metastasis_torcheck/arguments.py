from argparse import ArgumentParser, HelpFormatter


class SmartFormatter(HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            raw_text = text[2:]
            if "<br>" in raw_text:
                return raw_text.split('<br>')
            return HelpFormatter._split_lines(self, raw_text, width)
            # this is the RawTextHelpFormatter._split_lines
        return HelpFormatter._split_lines(self, text, width)


def get_args_for_egress():
    parser = ArgumentParser(
        prog='python main_egress.py',
        description='Filter tor clients traffic from certain users',
        formatter_class=SmartFormatter
    )
    parser.add_argument(
        '--peers-file',
        dest='peers',
        default=None,
        help='Path to file with known peers'
    )
    parser.add_argument(
        '--blacklist-file',
        dest='blacklist',
        default=None,
        help="""R|Path to file with blacklisted sites.
                        Each line in this file has to be formatted as this:<br><URL>;<IP address #1>,<IP #2>,<IP #N>
        """
    )
    return parser


def get_args_for_ingress():
    parser = ArgumentParser(
        prog='python main_ingress.py',
        description='Filter tor clients traffic from certain users'
    )
    parser.add_argument(
        '--clients-file',
        dest='clients',
        default=None,
        help="""
        Path to file with clients that has to be tracked.
        Traffic from these clients will be marked and analyzed against blacklist
        """
    )
    return parser
