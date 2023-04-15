from pathlib import Path

import paramiko
import os


def sshexec(ssh, cmd, with_printing=True):
    if with_printing:
        print(f'$ {cmd}')
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.set_combine_stderr(True)
    ret = []
    for line in iter(stdout.readline, ""):
        # print(line, end="")
        ret.append(line)
    if with_printing:
        from yachalk import chalk
        print(chalk.dim(''.join(ret)))

    # return ret


class SFTPClient(paramiko.SFTPClient):
    def __init__(self, *args, **kwargs):
        super(SFTPClient, self).__init__(*args, **kwargs)
        self.max_size = 1000000000
        self.urls = {
            'sd-v1-5.ckpt': '',
            'vae.vae.pt'  : '',
        }
        self.ssh = None
        self.enable_urls = True
        self.enable_print_upload = False
        self.enable_print_download = True
        self.rsync = True
        self.ip = None
        self.port = None
        self.print_rsync = True

    def put_any(self, source, target, forbid_rsync=False, forbid_recursive=False):
        source = Path(source)
        target = Path(target)

        if source.is_dir():
            self.mkdir(target.as_posix(), ignore_existing=True)
            self.put_dir(source.as_posix(), target.as_posix(), forbid_rsync=forbid_rsync, forbid_recursive=forbid_recursive)
        else:
            print(f"Uploading {source.as_posix()} to {target.as_posix()}")
            self.put(source.as_posix(), target.as_posix())

    def put_dir(self, source, target, *, forbid_rsync=False, ignore_exts=[], forbid_recursive=False):
        """
        Uploads the contents of the source directory to the target path. The
        target directory needs to exists. All subdirectories in source are
        created under target.
        """
        import yachalk as chalk
        print(f"{target} ({chalk.chalk.dim(target)})")

        if self.rsync and not forbid_rsync:
            flag = ''
            if self.print_rsync:
                flag = 'v'
            if not forbid_recursive:
                flag += 'r'

            cm = f"rsync -rlptgoDz{flag} -e 'ssh -p {self.port}' {source} root@{self.ip}:{Path(target).parent}"
            # print(cm)
            os.system(cm)
            return

        self.mkdir(target, ignore_existing=True)
        for item in os.listdir(source):
            if '.git' in item:
                continue

            if os.path.isfile(os.path.join(source, item)):
                if self.exists(os.path.join(target, item)):
                    continue

                # If size is above self.max_size (in bytes)
                if os.path.getsize(os.path.join(source, item)) > self.max_size:
                    if self.enable_urls and item in self.urls:
                        url = self.urls[item]
                        if url:
                            sshexec(self.ssh, f"wget {url} -O {os.path.join(target, item)}")
                            if self.enable_print_download:
                                self.print_download(item, url, target, url)
                            continue
                        else:
                            print(chalk.red("<!> Invalid URL '{url}' for "), item)

                    from yachalk import chalk
                    print(chalk.red("<!> File too big, skipping"), item)
                    continue

                if self.enable_print_upload:
                    self.print_upload(item, source, target)

                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item), forbid_rsync=forbid_rsync)

    def print_upload(self, item, source, target):
        print(f"Uploading {os.path.join(source, item)} to {target}")

    def print_download(self, item, source, target, url):
        print(f"Downloading {item} from {source} to {target}")

    def exists(self, path):
        try:
            # print(f'check if {path} exists', self.stat(path))
            if self.lstat(path) is not None:
                return True
            return False
        except:
            return False

    def mkdir(self, path, mode=511, ignore_existing=False):
        """
        Augments mkdir by adding an option to not fail if the folder exists
        """
        try:
            super(SFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise
