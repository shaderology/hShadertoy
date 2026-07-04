#!/usr/bin/env python
"""
OpenCL Kernel Compiler with Flexible File Composition

Usage examples:
    python compilecl.py main_kernel.cl
    python compilecl.py --header test_header.cl test_kernel.cl
    python compilecl.py -o custom_opts.json --header test_header.cl test_kernel.cl
    python compilecl.py --main-header my_header.cl --main-kernel my_kernel.cl test.cl
    
Kernel source construction order:
    1. main_header (main_header.cl by default)
    2. test header (if specified with --header)
    3. main_kernel (main_kernel.cl by default)
    4. test kernel (if specified as positional arg)
"""

import pyopencl as cl
import sys
import os
import json
import argparse


def load_build_options(json_file):
    """Load build options from JSON file and flatten to space-separated string."""
    if not json_file:
        return ""
        
    if not os.path.isfile(json_file):
        print(f"Warning: Build options file '{json_file}' not found, using empty options.")
        return ""
    
    try:
        with open(json_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return ""
            data = json.loads(content)
        
        opts = []
        
        def flatten_value(key, value):
            """Recursively flatten JSON values to compiler options."""
            if value is True:
                # Boolean true: add -D for defines
                if not key.startswith('-'):
                    return [f"-D{key}"]
                return [key]
            elif value is False or value is None:
                # Skip false/null values
                return []
            elif isinstance(value, (int, float)):
                # Numeric value: key=value
                if not key.startswith('-'):
                    return [f"-D{key}={value}"]
                return [f"{key}={value}"]
            elif isinstance(value, str):
                # String value
                if key.startswith('-I'):
                    # Include paths
                    return [f"-I{value}"]
                elif key.startswith('-D'):
                    # Define
                    return [f"{key}{value}"]
                elif key.startswith('-'):
                    # Other options
                    return [f"{key} {value}"]
                else:
                    # Regular define
                    return [f"-D{key}={value}"]
            elif isinstance(value, list):
                # List: expand each item with the key prefix if it's a flag
                result = []
                for item in value:
                    if isinstance(item, str):
                        if key.startswith('-'):
                            result.append(f"{key}{item}")
                        else:
                            result.append(item)
                    else:
                        result.append(str(item))
                return result
            elif isinstance(value, dict):
                # Nested dict: recurse
                result = []
                for k, v in value.items():
                    result.extend(flatten_value(k, v))
                return result
            else:
                return [str(value)]
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Skip metadata keys
                if key in ['comment', 'description', 'version']:
                    continue
                opts.extend(flatten_value(key, value))
        elif isinstance(data, list):
            # If list, join with spaces
            opts = [str(item) for item in data]
        elif isinstance(data, str):
            # If string, return as-is
            return data
        else:
            print(f"Warning: Unexpected JSON format in '{json_file}', using empty options.")
            return ""
        
        return ' '.join(opts)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{json_file}': {e}")
        return ""


def load_file_or_empty(filepath, description):
    """Load file content or return empty string with warning."""
    if filepath and os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    elif filepath:
        print(f"Warning: {description} file '{filepath}' not found, skipping.")
    return ""


def construct_kernel_source(main_header, test_header, main_kernel, test_kernel):
    """Construct kernel source by concatenating files in order."""
    sources = []
    
    # 1. Main header (required)
    main_header_src = load_file_or_empty(main_header, "Main header")
    if main_header_src:
        sources.append(f"// ============ Main Header: {main_header} ============\n")
        sources.append(main_header_src)
        sources.append("\n\n")
    
    # 2. Test header (optional)
    if test_header:
        test_header_src = load_file_or_empty(test_header, "Test header")
        if test_header_src:
            sources.append(f"// ============ Test Header: {test_header} ============\n")
            sources.append(test_header_src)
            sources.append("\n\n")
    
    # 3. Main kernel BEGIN (required)
    # kernel void generickernel(..){
    main_kernel_src = load_file_or_empty(main_kernel, "Main kernel")
    if main_kernel_src:
        sources.append(f"// ============ Main Kernel: {main_kernel} ============\n")
        sources.append(main_kernel_src)
        sources.append("\n\n")
    
    # 4. Test kernel code (optional)
    # Kernel code conained within main kernel
    if test_kernel:
        test_kernel_src = load_file_or_empty(test_kernel, "Test kernel")
        if test_kernel_src:
            sources.append(f"// ============ Test Kernel Code: {test_kernel} ============\n")
            sources.append(test_kernel_src)
            sources.append("\n\n")
    
    # 5. Main kernel END
    # }
    sources.append("AT_fragColor_set(fragColor);}")

    return ''.join(sources)


def enumerate_devices():
    """Enumerate all OpenCL platforms and devices."""
    platforms = cl.get_platforms()
    devices_list = []
    
    print("\nAvailable OpenCL devices:\n")
    for pi, plat in enumerate(platforms):
        print(f"[Platform {pi}] {plat.name}")
        for di, dev in enumerate(plat.get_devices()):
            dtype = cl.device_type.to_string(dev.type)
            idx = len(devices_list)
            devices_list.append((plat, dev))
            print(f"   [{idx}] {dev.name}  ({dtype})")
    print()
    
    return devices_list


def main():
    parser = argparse.ArgumentParser(
        description='OpenCL Kernel Compiler with flexible file composition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s main_kernel.cl
  %(prog)s --header test_header.cl test_kernel.cl
  %(prog)s -o custom_opts.json --header test_header.cl test_kernel.cl
  %(prog)s --main-header my_header.cl --main-kernel my_kernel.cl test.cl
  %(prog)s -d 1 --header test_header.cl test_kernel.cl

Kernel source construction order:
  1. main_header (main_header.cl by default)
  2. test header (if specified with --header)
  3. main_kernel (main_kernel.cl by default)
  4. test kernel (positional argument)
        """,
        add_help=False  # We'll add help manually to avoid conflict
    )
    
    # Required/positional arguments
    parser.add_argument(
        'test_kernel',
        nargs='?',
        default=None,
        help='Test kernel file to include (optional)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--build-opts',
        dest='build_opts',
        default='C:/dev/hShadertoy/tests/build_options.json',
        metavar='FILE',
        help='JSON file with build options (default: build_options.json)'
    )
    
    parser.add_argument(
        '--main-header',
        dest='main_header',
        default='C:/dev/hShadertoy/tests/ocl/main_header.cl',
        metavar='FILE',
        help='Main header file (default: main_header.cl)'
    )
    
    parser.add_argument(
        '--main-kernel',
        dest='main_kernel',
        default='C:/dev/hShadertoy/tests/ocl/main_kernel.cl',
        metavar='FILE',
        help='Main kernel file (default: main_kernel.cl)'
    )
    
    parser.add_argument(
        '--header',
        dest='test_header',
        default=None,
        metavar='FILE',
        help='Optional test header file'
    )
    
    parser.add_argument(
        '-d', '--device',
        dest='device',
        type=int,
        default=0,
        metavar='INDEX',
        help='OpenCL device index to use (default: 0)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output including full kernel source'
    )
    
    parser.add_argument(
        '--help',
        action='help',
        help='Show this help message and exit'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OpenCL Kernel Compiler")
    print("=" * 70)
    
    print("\nConfiguration:")
    print(f"  Main header    : {args.main_header}")
    print(f"  Test header    : {args.test_header or '(none)'}")
    print(f"  Main kernel    : {args.main_kernel}")
    print(f"  Test kernel    : {args.test_kernel or '(none)'}")
    print(f"  Build options  : {args.build_opts}")
    print(f"  Device index   : {args.device}")
    
    # Enumerate devices
    devices_list = enumerate_devices()
    
    if not devices_list:
        print("Error: No OpenCL devices found.")
        sys.exit(1)
    
    # Select device
    if args.device >= len(devices_list):
        print(f"Error: Device index {args.device} out of range (0-{len(devices_list)-1})")
        sys.exit(1)
    
    platform, device = devices_list[args.device]
    
    # Load build options
    build_opts = load_build_options(args.build_opts)
    
    # Construct kernel source
    kernel_src = construct_kernel_source(
        args.main_header,
        args.test_header,
        args.main_kernel,
        args.test_kernel
    )
    
    if not kernel_src.strip():
        print("Error: No kernel source code generated.")
        sys.exit(1)
    
    print(f"\nUsing platform: {platform.name}")
    print(f"Using device  : {device.name}")
    print(f"Build options : {build_opts or '(none)'}")
    print(f"\nKernel source length: {len(kernel_src)} characters")
    
    if args.verbose:
        print(f"\n{'=' * 70}")
        print("KERNEL SOURCE:")
        print(f"{'=' * 70}")
        print(kernel_src)
        print(f"{'=' * 70}\n")
    
    print(f"{'=' * 70}")
    print("Compiling...")
    print(f"{'=' * 70}\n")
    
    # Create context and compile
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    try:
        program = cl.Program(ctx, kernel_src).build(options=build_opts if build_opts else None)
        print("Kernel compiled successfully!\n")
        
        # List available kernels
        kernel_names = program.get_info(cl.program_info.KERNEL_NAMES).split(';')
        kernel_names = [k.strip() for k in kernel_names if k.strip()]
        
        if kernel_names:
            print(f"Available kernels ({len(kernel_names)}):")
            for kname in kernel_names:
                print(f"  • {kname}")
        print()
        
    except cl.RuntimeError as e:
        # pyopencl has no BuildError; build failures raise cl.RuntimeError
        # (BUILD_PROGRAM_FAILURE) with the device build log in the message.
        print("Build failed!\n")
        print("=" * 70)
        print(str(e))
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
