import paramiko
import os

def sshexec(ssh, cmd, with_printing=True):
    if with_printing:
        print(f'sshexec({cmd})')
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

    def put_dir(self, source, target, ignore_exts=[]):
        """
        Uploads the contents of the source directory to the target path. The
        target directory needs to exists. All subdirectories in source are
        created under target.
        """
        print(source, target)
        self.mkdir(target, ignore_existing=True)
        for item in os.listdir(source):
            if '.git' in item:
                continue

            if os.path.isfile(os.path.join(source, item)):
                # If size is above self.max_size (in bytes)
                if os.path.getsize(os.path.join(source, item)) > self.max_size:
                    if item in self.urls:
                        url = self.urls[item]
                        sshexec(self.ssh, f"wget {url} -O {os.path.join(source, item)}")
                        continue

                    from yachalk import chalk
                    print(chalk.red("<!> File too big, skipping"), item)
                    continue

                print("Uploading %s to %s" % (os.path.join(source, item), target))
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def exists(self, path):
        try:
            self.stat(path)
            return True
        except:
            return False

    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(SFTPClient, self).mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise
