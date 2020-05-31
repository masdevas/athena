import argparse
import os


def bin2c(input):
  with open(input, 'rb') as in_file:
        data = in_file.read()

  base=os.path.basename(input)
  varname = base.split('.')[0]
  linesize = 80
  indent = 2
  byte_len = 6  # '0x00, '

  out = 'std::array<uint8_t, %d> %s = {\n' % (len(data), varname)
  line = ''
  for byte in data:
    line += '0x%02x, ' % byte
    if len(line) + indent + byte_len >= linesize:
      out += ' ' * indent + line + '\n'
      line = ''
  if len(line) + indent + byte_len < linesize:
    out += ' ' * indent + line + '\n'
  # strip the last comma
  out = out.rstrip(', \n') + '\n'
  out += '};\n'
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--header', type=str)
  parser.add_argument('input', action='append')

  args = parser.parse_args()

  header = '#include <array>\n#pragma once\n\n'

  for file in args.input:
    header += bin2c(file)
  
  with open(args.header, 'w') as outfile:
    outfile.write(header)

if __name__ == "__main__":
  main()
