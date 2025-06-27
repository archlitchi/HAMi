/*
Copyright 2024 The HAMi Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package nvidia

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/Project-HAMi/HAMi/pkg/device/nvidia"
	v0 "github.com/Project-HAMi/HAMi/pkg/monitor/nvidia/v0"
	v1 "github.com/Project-HAMi/HAMi/pkg/monitor/nvidia/v1"
	"github.com/Project-HAMi/HAMi/pkg/util"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
	listerscorev1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
)

const SharedRegionMagicFlag = 19920718

type headerT struct {
	initializedFlag int32
	majorVersion    int32
	minorVersion    int32
}

type UsageInfo interface {
	DeviceMax() int
	DeviceNum() int
	DeviceMemoryContextSize(idx int) uint64
	DeviceMemoryModuleSize(idx int) uint64
	DeviceMemoryBufferSize(idx int) uint64
	DeviceMemoryOffset(idx int) uint64
	DeviceMemoryTotal(idx int) uint64
	DeviceSmUtil(idx int) uint64
	SetDeviceSmLimit(l uint64)
	IsValidUUID(idx int) bool
	DeviceUUID(idx int) string
	DeviceMemoryLimit(idx int) uint64
	SetDeviceMemoryLimit(l uint64)
	LastKernelTime() int64
	//UsedMemory(idx int) (uint64, error)
	GetPriority() int
	GetRecentKernel() int32
	SetMemoryUpdate(v uint64)
	GetMemoryUpdate() uint64
	SetRecentKernel(v int32)
	GetUtilizationSwitch() int32
	SetUtilizationSwitch(v int32)
}

type ContainerUsage struct {
	Pod           *corev1.Pod
	PodUID        string
	ContainerName string
	data          []byte
	Info          UsageInfo
}

type ContainerLister struct {
	containerPath string
	PodLister     listerscorev1.PodLister
	containers    map[string]*ContainerUsage
	mutex         sync.Mutex
	clientset     *kubernetes.Clientset
}

func NewContainerLister() (*ContainerLister, error) {
	util.SupportDevices[nvidia.NvidiaGPUDevice] = "hami.io/vgpu-devices-allocated"
	hookPath, ok := os.LookupEnv("HOOK_PATH")
	if !ok {
		return nil, fmt.Errorf("HOOK_PATH not set")
	}
	config, err := clientcmd.BuildConfigFromFlags("", os.Getenv("KUBECONFIG"))
	if err != nil {
		klog.Errorf("Failed to build kubeconfig: %v", err)
		return nil, err
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		klog.Errorf("Failed to build clientset: %v", err)
		return nil, err
	}
	return &ContainerLister{
		containerPath: filepath.Join(hookPath, "containers"),
		containers:    make(map[string]*ContainerUsage),
		clientset:     clientset,
	}, nil
}

func (l *ContainerLister) Lock() {
	l.mutex.Lock()
}

func (l *ContainerLister) UnLock() {
	l.mutex.Unlock()
}

func (l *ContainerLister) ListContainers() map[string]*ContainerUsage {
	return l.containers
}

func (l *ContainerLister) Clientset() *kubernetes.Clientset {
	return l.clientset
}

func (l *ContainerLister) Update() error {
	klog.Infoln("into Updating......")
	nodename := os.Getenv(util.NodeNameEnvName)
	if nodename == "" {
		return fmt.Errorf("env %s not set", util.NodeNameEnvName)
	}
	pods, err := l.PodLister.List(labels.SelectorFromSet(labels.Set{util.AssignedNodeAnnotations: nodename}))
	if err != nil {
		klog.Errorf("Failed to list pods for node %s: %v", nodename, err)
		return fmt.Errorf("failed to list pods: %w", err)
	}

	l.mutex.Lock()
	defer l.mutex.Unlock()
	entries, err := os.ReadDir(l.containerPath)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dirName := filepath.Join(l.containerPath, entry.Name())
		pod := isValidPod(entry.Name(), pods)
		if pod == nil {
			dirInfo, err := os.Stat(dirName)
			if err == nil && dirInfo.ModTime().Add(time.Second*300).After(time.Now()) {
				continue
			}
			klog.Infof("Removing dirname %s in monitorpath", dirName)
			if c, ok := l.containers[entry.Name()]; ok {
				syscall.Munmap(c.data)
				delete(l.containers, entry.Name())
			}
			_ = os.RemoveAll(dirName)
			continue
		}
		if data, ok := l.containers[entry.Name()]; ok {
			if data != nil && data.Info.DeviceMemoryLimit(0) != 0 {
				podDev, _ := util.DecodePodDevices(util.SupportDevices, pod.Annotations)
				for ctr, val := range podDev[nvidia.NvidiaGPUDevice] {
					if pod.Spec.Containers[ctr].Name != data.ContainerName {
						continue
					}
					klog.Infoln("found container name=", pod.Spec.Containers[ctr].Name)
					devicemap := map[string]uint64{}
					num := data.Info.DeviceNum()
					for i := 0; i < num; i++ {
						devicemap[data.Info.DeviceUUID(i)[:40]] = data.Info.DeviceMemoryLimit(i)
					}
					for _, ctrdevs := range val {
						if devicemap[ctrdevs.UUID] != uint64(ctrdevs.Usedmem)*1024*1024 {
							klog.Infoln("setting ctrdevices from", devicemap[ctrdevs.UUID], "to", ctrdevs.Usedmem*1024*1024)
							data.Info.SetDeviceMemoryLimit(uint64(ctrdevs.Usedmem) * 1024 * 1024)
						}
					}
				}
			}
			continue
		}
		usage, err := loadCache(dirName)
		if err != nil {
			klog.Errorf("Failed to load cache: %s, error: %v", dirName, err)
			continue
		}
		if usage == nil {
			// no cuInit in container
			continue
		}
		usage.PodUID = strings.Split(entry.Name(), "_")[0]
		usage.Pod = pod
		usage.ContainerName = strings.Split(entry.Name(), "_")[1]
		l.containers[entry.Name()] = usage
		klog.Infof("Adding ctr dirname %s in monitorpath", dirName)
	}
	return nil
}

func loadCache(fpath string) (*ContainerUsage, error) {
	klog.Infof("Checking path %s", fpath)
	files, err := os.ReadDir(fpath)
	if err != nil {
		return nil, err
	}
	if len(files) > 2 {
		return nil, errors.New("cache num not matched")
	}
	if len(files) == 0 {
		return nil, nil
	}
	cacheFile := ""
	for _, val := range files {
		if strings.Contains(val.Name(), "libvgpu.so") {
			continue
		}
		if !strings.Contains(val.Name(), ".cache") {
			continue
		}
		cacheFile = filepath.Join(fpath, val.Name())
		break
	}
	if cacheFile == "" {
		klog.Infof("No cache file in %s", fpath)
		return nil, nil
	}
	info, err := os.Stat(cacheFile)
	if err != nil {
		klog.Errorf("Failed to stat cache file: %s, error: %v", cacheFile, err)
		return nil, err
	}
	if info.Size() < int64(unsafe.Sizeof(headerT{})) {
		return nil, fmt.Errorf("cache file size %d too small", info.Size())
	}
	f, err := os.OpenFile(cacheFile, os.O_RDWR, 0666)
	if err != nil {
		klog.Errorf("Failed to open cache file: %s, error: %v", cacheFile, err)
		return nil, err
	}
	defer func(f *os.File) {
		_ = f.Close()
	}(f)
	usage := &ContainerUsage{}
	usage.data, err = syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_WRITE|syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		klog.Errorf("Failed to mmap cache file: %s, error: %v", cacheFile, err)
		return nil, err
	}
	head := (*headerT)(unsafe.Pointer(&usage.data[0]))
	if head.initializedFlag != SharedRegionMagicFlag {
		_ = syscall.Munmap(usage.data)
		return nil, fmt.Errorf("cache file magic flag not matched")
	}
	if info.Size() == 1197897 {
		usage.Info = v0.CastSpec(usage.data)
	} else if head.majorVersion == 1 {
		usage.Info = v1.CastSpec(usage.data)
	} else {
		_ = syscall.Munmap(usage.data)
		return nil, fmt.Errorf("unknown cache file size %d version %d.%d", info.Size(), head.majorVersion, head.minorVersion)
	}
	return usage, nil
}

func isValidPod(name string, pods []*corev1.Pod) *corev1.Pod {
	for _, val := range pods {
		if strings.Contains(name, string(val.UID)) {
			return val
		}
	}
	return nil
}
